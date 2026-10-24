import os, json, uuid, hashlib, sqlite3, threading, re, time
from datetime import datetime, timezone
from typing import Any
from fastapi import FastAPI, Request, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx

APP_TITLE = "A2A Invoice Action Agent"
DB_PATH = os.getenv("DB_PATH", "/tmp/invoice_agent.db")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")  # Optional shared-token mode; otherwise bearer tokens identify isolated principals.
AI_API_KEY = os.getenv("GROQ_API_KEY", "")
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")
AI_MODEL = os.getenv("AI_MODEL", "llama-3.1-8b-instant")
PREFIX = "/a2a"
PROPOSAL_MT = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPT_MT = "application/vnd.ga5.invoice-action-receipts+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
RESULT_MT = "application/vnd.ga5.invoice-action-results+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}
db_lock = threading.RLock()

app = FastAPI(title=APP_TITLE, docs_url=None, redoc_url=None)

def connect():
    c = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    with connect() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS tasks (
          id TEXT PRIMARY KEY, principal TEXT NOT NULL, context_id TEXT NOT NULL,
          batch_id TEXT NOT NULL, state TEXT NOT NULL, task_json TEXT NOT NULL,
          proposals_json TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS messages (
          principal TEXT NOT NULL, message_id TEXT NOT NULL, digest TEXT NOT NULL,
          task_id TEXT NOT NULL, response_json TEXT NOT NULL,
          PRIMARY KEY(principal,message_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS decisions (
          package_hash TEXT PRIMARY KEY, proposal_json TEXT NOT NULL, created REAL NOT NULL)""")
        c.commit()
init_db()

def now_iso():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def principal_from(auth):
    if not auth or not auth.startswith("Bearer ") or not auth[7:].strip():
        raise HTTPException(401, "UNAUTHENTICATED")
    token = auth[7:].strip()
    if AUTH_TOKEN and token != AUTH_TOKEN:
        raise HTTPException(403, "FORBIDDEN")
    return hashlib.sha256(token.encode()).hexdigest()

def check_headers(request: Request, content_type_required=False):
    if request.headers.get("A2A-Version") != "1.0":
        raise HTTPException(400, "VERSION_NOT_SUPPORTED")
    if content_type_required:
        ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if ct != "application/a2a+json":
            raise HTTPException(415, "CONTENT_TYPE_NOT_SUPPORTED")

def task_status(state):
    return {"state": state}

def make_part(mt, data):
    return {"mediaType": mt, "data": data}

def make_task(task_id, context_id, state, history=None, artifacts=None, status_message=None):
    result = {"id": task_id, "contextId": context_id, "status": {"state": state},
              "history": history or [], "artifacts": artifacts or []}
    if status_message:
        result["status"]["message"] = status_message
    return result

def store_task(task, principal, batch_id, proposals):
    t = time.time()
    with connect() as c:
        c.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?,?,?,?)",
                  (task["id"], principal, task["contextId"], batch_id,
                   task["status"]["state"], canonical(task), canonical(proposals), t, t))
        c.commit()

def load_task(task_id, principal):
    with connect() as c:
        row = c.execute("SELECT * FROM tasks WHERE id=? AND principal=?", (task_id, principal)).fetchone()
    if not row:
        # Generic response: do not reveal whether the task exists for another principal.
        raise HTTPException(404, "TASK_NOT_FOUND")
    return row

def save_task_json(task_id, principal, task, state, proposals=None):
    with connect() as c:
        if proposals is None:
            c.execute("UPDATE tasks SET state=?, task_json=?, updated=? WHERE id=? AND principal=?",
                      (state, canonical(task), time.time(), task_id, principal))
        else:
            c.execute("UPDATE tasks SET state=?, task_json=?, proposals_json=?, updated=? WHERE id=? AND principal=?",
                      (state, canonical(task), canonical(proposals), time.time(), task_id, principal))
        c.commit()

def get_base(request):
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL.rstrip("/") + PREFIX + "/"
    # Deploy at the domain root by default; set PUBLIC_BASE_URL to the origin if using a subpath.
    return str(request.base_url).rstrip("/") + PREFIX + "/"

@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base = get_base(request)
    return {
      "name": "Invoice Action Agent",
      "description": "Reviews invoice packages, proposes evidence-backed actions, and executes only accepted proposals.",
      "version": "1.0.0",
      "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
      "capabilities": {"streaming": False, "pushNotifications": False},
      "defaultInputModes": [BATCH_MT, RESULT_MT],
      "defaultOutputModes": [PROPOSAL_MT, RECEIPT_MT],
      "skills": [{"id": "invoice_action_agent", "name": "Invoice Action Agent",
                  "description": "Classifies invoice packages and safely records accepted actions.",
                  "tags": ["invoice", "approval", "duplicate-payment", "evidence"]}]
    }

def find_refs(text):
    return list(dict.fromkeys(re.findall(r"\[[^\[\]\n]{1,100}\]", text)))

def flatten_package(package):
    # Preserve source content verbatim for the AI. Documents may be represented in several common shapes.
    return canonical(package)

def extract_package_identity(package):
    pid = package.get("packageId") or package.get("id") or package.get("package_id")
    return str(pid) if pid is not None else ""

def heuristic_decision(package):
    """Conservative fallback. For high accuracy, configure GROQ_API_KEY so the documents are reviewed by an LLM."""
    raw = flatten_package(package)
    low = raw.lower()
    refs = find_refs(raw)
    action = "open_exception"
    if any(x in low for x in ["already paid", "payment completed", "paid previously", "duplicate invoice", "same invoice was paid"]):
        action = "reject_duplicate"
    elif any(x in low for x in ["verify bank details", "verification pending", "awaiting verification", "hold payment", "do not pay until", "do not release until"]):
        action = "hold_invoice"
    elif any(x in low for x in ["exceeds delegated", "above approval limit", "requires approval", "approval required", "outside delegated authority"]):
        action = "request_approval"
    elif any(x in low for x in ["reconciled", "goods received", "within authority", "within autonomous", "approved for payment"]) and not any(x in low for x in ["not reconciled", "not approved", "not received"]):
        action = "settle_invoice"
    # Basic best-effort fact extraction from common JSON fields.
    def get_any(keys, default="UNKNOWN"):
        for k in keys:
            if package.get(k) not in (None, ""):
                return package[k]
        return default
    vendor = get_any(["vendorName", "vendor", "supplierName", "supplier"])
    invoice = get_any(["invoiceNumber", "invoiceNo", "invoice_number", "number"])
    amount = get_any(["amountMinor", "amount_minor"])
    if amount == "UNKNOWN":
        amount_value = get_any(["amount", "total", "invoiceTotal"], 0)
        try:
            amount = int(round(float(str(amount_value).replace(",", "").replace("₹", "").replace("INR", "").strip()) * 100))
        except Exception:
            amount = 0
    try: amount = int(amount)
    except Exception: amount = 0
    currency = get_any(["currency"], "INR")
    # The exact decisive references should come from the determining paragraph, not the whole package.
    # The fallback limits refs to the final three in the source to avoid including cover/archive refs.
    evidence = refs[-3:] if len(refs) >= 3 else refs
    if len(evidence) != 3:
        evidence = (evidence + ["[evidence-not-found]"] * 3)[:3]
    rationale = (f"Proposed action: {action}. The available package text suggests this action based on evidence "
                 f"{evidence[0]} and {evidence[1]}; {evidence[2]} is the third cited reference. "
                 "This fallback is conservative and should be replaced by an AI-reviewed decision when document structure is complex.")
    return {"packageId": extract_package_identity(package), "actionId": uuid.uuid4().hex[:16],
            "action": action, "facts": {"vendorName": str(vendor), "invoiceNumber": str(invoice),
            "amountMinor": amount, "currency": str(currency)}, "evidenceRefs": evidence, "rationale": rationale}

async def ai_decisions(packages):
    # One batched call for all uncached packages. Output is schema-validated before persistence.
    if not AI_API_KEY:
        return [heuristic_decision(p) for p in packages]
    system = """You are a cautious accounts-payable invoice decision engine. Analyze each package's documents, ignoring archive examples, training decoys, old examples, and irrelevant action words. Choose exactly one action:
settle_invoice = valid, reconciled, within autonomous authority;
request_approval = commercially valid but outside delegated authority;
hold_invoice = payment pauses until a stated verification completes;
reject_duplicate = the same commercial invoice was already paid (different invoice is not duplicate);
open_exception = material records conflict and need exception workflow.
Return ONLY JSON: {"proposals":[{"packageId":"...","action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR"},"evidenceRefs":["exact ref","exact ref","exact ref"],"rationale":"60-1500 chars citing at least two refs"}]}. Use exactly the three decisive bracketed references from the paragraph determining the action. No cover-sheet, archive-example, or decoy refs. AmountMinor is integer minor currency units."""
    prompt = {"packages": packages}
    async with httpx.AsyncClient(timeout=35) as client:
        r = await client.post(AI_BASE_URL, headers={"Authorization": f"Bearer {AI_API_KEY}",
            "Content-Type": "application/json"}, json={"model": AI_MODEL, "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": canonical(prompt)}]})
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    parsed = json.loads(content)
    proposals = parsed.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != len(packages):
        raise ValueError("AI returned wrong proposal count")
    by_id = {str(p.get("packageId")): p for p in proposals}
    clean = []
    for package in packages:
        pid = extract_package_identity(package)
        p = by_id.get(pid)
        if not p or p.get("action") not in ACTIONS:
            raise ValueError("AI returned invalid proposal")
        facts = p.get("facts") or {}
        refs = p.get("evidenceRefs")
        rationale = p.get("rationale")
        if not isinstance(refs, list) or len(refs) != 3 or not all(isinstance(x, str) and x.startswith("[") and x.endswith("]") for x in refs):
            raise ValueError("AI returned invalid evidence refs")
        if not isinstance(rationale, str) or not (60 <= len(rationale) <= 1500):
            raise ValueError("AI returned invalid rationale")
        try: amount = int(facts["amountMinor"])
        except Exception: raise ValueError("AI returned invalid amountMinor")
        clean.append({"packageId": pid, "actionId": uuid.uuid4().hex + uuid.uuid4().hex[:4],
          "action": p["action"], "facts": {"vendorName": str(facts["vendorName"]),
          "invoiceNumber": str(facts["invoiceNumber"]), "amountMinor": amount,
          "currency": str(facts["currency"])}, "evidenceRefs": refs, "rationale": rationale})
    return clean

async def cached_decisions(packages):
    results, missing = {}, []
    with connect() as c:
        for p in packages:
            h = digest(p)
            row = c.execute("SELECT proposal_json FROM decisions WHERE package_hash=?", (h,)).fetchone()
            if row:
                results[h] = json.loads(row["proposal_json"])
            else:
                missing.append((h, p))
    if missing:
        fresh = await ai_decisions([p for _, p in missing])
        with connect() as c:
            for (h, package), proposal in zip(missing, fresh):
                # Store semantic decision independent of task/batch. Action ID is regenerated per task.
                semantic = {k:v for k,v in proposal.items() if k != "actionId"}
                c.execute("INSERT OR REPLACE INTO decisions VALUES(?,?,?)", (h, canonical(semantic), time.time()))
                results[h] = semantic
            c.commit()
    output = []
    for p in packages:
        semantic = dict(results[digest(p)])
        semantic["actionId"] = uuid.uuid4().hex + uuid.uuid4().hex[:4]
        output.append(semantic)
    return output

def get_message_and_data(body):
    msg = body.get("message")
    if not isinstance(msg, dict):
        raise HTTPException(400, "INVALID_MESSAGE")
    if msg.get("role") != "ROLE_USER" or not msg.get("messageId"):
        raise HTTPException(400, "INVALID_MESSAGE")
    parts = msg.get("parts")
    if not isinstance(parts, list) or len(parts) != 1 or not isinstance(parts[0], dict) or not isinstance(parts[0].get("data"), dict):
        raise HTTPException(400, "INVALID_PARTS")
    return msg, parts[0]["mediaType"], parts[0]["data"]

@app.post(PREFIX + "/message:send")
async def message_send(request: Request, authorization: str | None = Header(default=None)):
    check_headers(request, True)
    principal = principal_from(authorization)
    try: body = await request.json()
    except Exception: raise HTTPException(400, "INVALID_JSON")
    msg, media_type, data = get_message_and_data(body)
    mid = str(msg["messageId"])
    msg_digest = digest(msg)  # Configuration intentionally excluded.
    with db_lock:
        with connect() as c:
            old = c.execute("SELECT digest,response_json FROM messages WHERE principal=? AND message_id=?", (principal, mid)).fetchone()
            if old:
                if old["digest"] != msg_digest:
                    raise HTTPException(409, "IDEMPOTENCY_CONFLICT")
                return JSONResponse(json.loads(old["response_json"]), media_type="application/a2a+json")
    task_id = str(uuid.uuid4())
    if media_type == BATCH_MT:
        batch_id = str(data.get("batchId", ""))
        packages = data.get("packages")
        if not batch_id or not isinstance(packages, list) or not packages:
            raise HTTPException(400, "INVALID_BATCH")
        ids = [extract_package_identity(p) for p in packages]
        if any(not x for x in ids) or len(ids) != len(set(ids)):
            raise HTTPException(400, "INVALID_PACKAGE_IDS")
        proposals = await cached_decisions(packages)
        if len({p["actionId"] for p in proposals}) != len(proposals):
            raise HTTPException(500, "ACTION_ID_COLLISION")
        artifact = {"artifactId": str(uuid.uuid4()), "name": "Invoice action proposals",
                    "parts": [make_part(PROPOSAL_MT, {"batchId": batch_id, "proposals": proposals})]}
        task = make_task(task_id, str(uuid.uuid4()), "TASK_STATE_INPUT_REQUIRED",
                         history=[msg], artifacts=[artifact])
        with db_lock:
            store_task(task, principal, batch_id, proposals)
            response = {"task": task}
            with connect() as c:
                c.execute("INSERT INTO messages VALUES(?,?,?,?,?)",
                          (principal, mid, msg_digest, task_id, canonical(response)))
                c.commit()
        return JSONResponse(response, media_type="application/a2a+json")
    elif media_type == RESULT_MT:
        task_id_in = msg.get("taskId")
        context_id = msg.get("contextId")
        if not task_id_in or not context_id:
            raise HTTPException(400, "INVALID_CONTINUATION")
        row = load_task(str(task_id_in), principal)
        with db_lock:
            task = json.loads(row["task_json"])
            if task["contextId"] != context_id or row["batch_id"] != str(data.get("batchId")):
                raise HTTPException(409, "TASK_CONTEXT_BATCH_MISMATCH")
            if row["state"] != "TASK_STATE_INPUT_REQUIRED":
                # Terminal replay of a new message ID returns the already-final Task without mutation.
                if row["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
                    return JSONResponse({"task": task}, media_type="application/a2a+json")
                raise HTTPException(409, "TASK_NOT_AWAITING_RESULTS")
            proposals = json.loads(row["proposals_json"])
            by_key = {(p["packageId"], p["actionId"], p["action"]): p for p in proposals}
            results = data.get("results")
            if not isinstance(results, list):
                raise HTTPException(400, "INVALID_RESULTS")
            executions, seen = [], set()
            for res in results:
                key = (str(res.get("packageId")), str(res.get("actionId")), str(res.get("action")))
                if key not in by_key or key in seen or res.get("outcome") not in ("ACCEPTED", "REJECTED") or not isinstance(res.get("receiptNonce"), str) or not res["receiptNonce"]:
                    raise HTTPException(409, "RESULT_DOES_NOT_MATCH_PROPOSAL")
                seen.add(key)
                if res["outcome"] == "ACCEPTED":
                    p = by_key[key]
                    executions.append({"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                      "receiptNonce": res["receiptNonce"], "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]})
            if len(seen) != len(proposals):
                raise HTTPException(409, "INCOMPLETE_RESULTS")
            # Atomic state transition under lock: cancel and result cannot both win.
            task["history"].append(msg)
            task["status"] = {"state": "TASK_STATE_COMPLETED"}
            task["artifacts"].append({"artifactId": str(uuid.uuid4()), "name": "Invoice action receipts",
                 "parts": [make_part(RECEIPT_MT, {"batchId": row["batch_id"], "executions": executions})]})
            save_task_json(task_id_in, principal, task, "TASK_STATE_COMPLETED")
            response = {"task": task}
            with connect() as c:
                c.execute("INSERT INTO messages VALUES(?,?,?,?,?)",
                          (principal, mid, msg_digest, task_id_in, canonical(response)))
                c.commit()
            return JSONResponse(response, media_type="application/a2a+json")
    else:
        raise HTTPException(415, "UNSUPPORTED_INPUT_MEDIA_TYPE")

@app.get(PREFIX + "/tasks")
async def list_tasks(request: Request, authorization: str | None = Header(default=None)):
    check_headers(request)
    principal = principal_from(authorization)
    with connect() as c:
        rows = c.execute("SELECT task_json FROM tasks WHERE principal=? ORDER BY created DESC", (principal,)).fetchall()
    return JSONResponse({"tasks": [json.loads(r["task_json"]) for r in rows]}, media_type="application/a2a+json")

@app.get(PREFIX + "/tasks/{task_id}")
async def read_task(task_id: str, request: Request, authorization: str | None = Header(default=None)):
    check_headers(request)
    principal = principal_from(authorization)
    row = load_task(task_id, principal)
    return JSONResponse(json.loads(row["task_json"]), media_type="application/a2a+json")

@app.post(PREFIX + "/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request, authorization: str | None = Header(default=None)):
    check_headers(request, True)
    principal = principal_from(authorization)
    with db_lock:
        row = load_task(task_id, principal)
        task = json.loads(row["task_json"])
        if row["state"] in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED"):
            raise HTTPException(409, "TASK_TERMINAL")
        task["status"] = {"state": "TASK_STATE_CANCELED"}
        save_task_json(task_id, principal, task, "TASK_STATE_CANCELED")
        return JSONResponse(task, media_type="application/a2a+json")

@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code,
                        media_type="application/a2a+json")

@app.get("/")
async def health():
    return {"service": APP_TITLE, "status": "ok"}
