import * as fs from "node:fs";
import * as path from "node:path";

// ── Config ────────────────────────────────────────────────────────────────────

const SUBREDDIT = "srilanka";
const TARGET = 50;
const DELAY_MS = 600;
const DATA_DIR = path.resolve("data");
const USER_AGENT = "sri-lanka-eda-research/1.0";
const BASE_URL = "https://api.pullpush.io/reddit/search";

// ── Types ─────────────────────────────────────────────────────────────────────

interface PullPushPost {
  id: string;
  title: string;
  selftext: string;
  author: string;
  score: number;
  upvote_ratio: number;
  num_comments: number;
  created_utc: number;
  permalink: string;
  url: string;
  link_flair_text: string | null;
  is_self: boolean;
  subreddit: string;
}

interface PullPushComment {
  id: string;
  body: string;
  author: string;
  score: number;
  created_utc: number;
  permalink: string;
  parent_id: string;
  link_id: string;
  subreddit: string;
}

interface PullPushResponse<T> {
  data: T[];
}

interface Entry {
  entry_id: string;
  type: "post" | "comment";
  post_id: string;
  title: string;
  text: string;
  author: string;
  score: number;
  upvote_ratio: number;
  num_comments: number;
  created_utc: number;
  created_date: string;
  permalink: string;
  flair: string;
  is_self: boolean;
}

// ── HTTP ──────────────────────────────────────────────────────────────────────

async function get<T>(url: string): Promise<T> {
  const resp = await fetch(url, {
    headers: { "User-Agent": USER_AGENT },
  });

  if (resp.status === 429) {
    const wait = Number(resp.headers.get("Retry-After") ?? 30) * 1000;
    statusLine(`rate-limited — waiting ${wait / 1000}s…`);
    await sleep(wait);
    return get(url);
  }

  if (resp.status === 503 || resp.status === 502) {
    statusLine(`server error ${resp.status} — retrying in 10s…`);
    await sleep(10_000);
    return get(url);
  }

  if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
  return resp.json() as Promise<T>;
}

// Walk backwards in time using `before` epoch cursor
async function* walkBack<T extends { created_utc: number }>(
  type: "submission" | "comment",
): AsyncGenerator<T[]> {
  // Start from now and walk backwards
  let before = Math.floor(Date.now() / 1000);

  while (true) {
    const url =
      `${BASE_URL}/${type}/?` +
      `subreddit=${SUBREDDIT}&limit=100&before=${before}&sort=desc`;

    let result: PullPushResponse<T>;
    try {
      result = await get<PullPushResponse<T>>(url);
    } catch (err) {
      statusLine(`error: ${err} — skipping`);
      break;
    }

    const items = result.data;
    if (items.length === 0) break;

    yield items;

    // Move cursor to just before the oldest item in this batch
    before = Math.min(...items.map((i) => i.created_utc)) - 1;

    await sleep(DELAY_MS);
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

// ── Data mapping ──────────────────────────────────────────────────────────────

function fromPost(p: PullPushPost): Entry {
  const text = [p.title, p.selftext]
    .map((s) => (s ?? "").trim())
    .filter(Boolean)
    .join("\n");

  return {
    entry_id: `t3_${p.id}`,
    type: "post",
    post_id: p.id,
    title: p.title ?? "",
    text,
    author: p.author ?? "[deleted]",
    score: p.score ?? 0,
    upvote_ratio: p.upvote_ratio ?? 0,
    num_comments: p.num_comments ?? 0,
    created_utc: p.created_utc,
    created_date: new Date(p.created_utc * 1000).toISOString(),
    permalink: p.permalink
      ? `https://reddit.com${p.permalink}`
      : `https://reddit.com/r/${SUBREDDIT}/comments/${p.id}`,
    flair: p.link_flair_text ?? "",
    is_self: p.is_self ?? false,
  };
}

function fromComment(c: PullPushComment): Entry {
  return {
    entry_id: `t1_${c.id}`,
    type: "comment",
    post_id: (c.link_id ?? "").replace("t3_", ""),
    title: "",
    text: c.body ?? "",
    author: c.author ?? "[deleted]",
    score: c.score ?? 0,
    upvote_ratio: 0,
    num_comments: 0,
    created_utc: c.created_utc,
    created_date: new Date(c.created_utc * 1000).toISOString(),
    permalink: c.permalink ? `https://reddit.com${c.permalink}` : "",
    flair: "",
    is_self: false,
  };
}

function isUsable(e: Entry): boolean {
  const t = e.text.trim();
  return (
    t.length > 0 &&
    t !== "[deleted]" &&
    t !== "[removed]" &&
    e.author !== "[deleted]"
  );
}

// ── CSV ───────────────────────────────────────────────────────────────────────

const CSV_COLS: (keyof Entry)[] = [
  "entry_id",
  "type",
  "post_id",
  "title",
  "text",
  "author",
  "score",
  "upvote_ratio",
  "num_comments",
  "created_utc",
  "created_date",
  "permalink",
  "flair",
  "is_self",
];

function csvCell(v: unknown): string {
  const s = String(v ?? "")
    .replace(/\r\n|\r/g, " ")
    .replace(/\n/g, " ");
  return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
}

function toCsvRow(e: Entry): string {
  return CSV_COLS.map((k) => csvCell(e[k])).join(",");
}

// ── Output / checkpoint ───────────────────────────────────────────────────────

let csvFd: number;
let jsonlFd: number;
let seenIds: Set<string>;
let written: number;

const CSV_PATH = path.join(DATA_DIR, "entries.csv");
const JSONL_PATH = path.join(DATA_DIR, "entries.jsonl");
const SEEN_PATH = path.join(DATA_DIR, "seen_ids.json");

function initOutput(): void {
  fs.mkdirSync(DATA_DIR, { recursive: true });

  try {
    seenIds = new Set(
      JSON.parse(fs.readFileSync(SEEN_PATH, "utf8")) as string[],
    );
    written = seenIds.size;
    console.log(
      `Resuming — ${written.toLocaleString()} entries already saved.`,
    );
  } catch {
    seenIds = new Set();
    written = 0;
    fs.writeFileSync(CSV_PATH, CSV_COLS.join(",") + "\n", "utf8");
    fs.writeFileSync(JSONL_PATH, "", "utf8");
  }

  csvFd = fs.openSync(CSV_PATH, "a");
  jsonlFd = fs.openSync(JSONL_PATH, "a");
}

function saveEntry(e: Entry): boolean {
  if (seenIds.has(e.entry_id) || !isUsable(e)) return false;
  fs.writeSync(csvFd, toCsvRow(e) + "\n");
  fs.writeSync(jsonlFd, JSON.stringify(e) + "\n");
  seenIds.add(e.entry_id);
  written++;
  return true;
}

function checkpoint(): void {
  fs.writeFileSync(SEEN_PATH, JSON.stringify([...seenIds]), "utf8");
}

function closeOutput(): void {
  try {
    fs.closeSync(csvFd);
  } catch {}
  try {
    fs.closeSync(jsonlFd);
  } catch {}
  checkpoint();
}

// ── Progress ──────────────────────────────────────────────────────────────────

function statusLine(msg: string): void {
  const pct = Math.min(100, (written / TARGET) * 100).toFixed(1);
  process.stdout.write(
    `\r[${pct.padStart(5)}%] ${written.toLocaleString().padStart(6)} / ${TARGET.toLocaleString()} | ${msg}          `,
  );
}

// ── Collection ────────────────────────────────────────────────────────────────

async function collectPosts(): Promise<void> {
  let batches = 0;
  for await (const batch of walkBack<PullPushPost>("submission")) {
    let added = 0;
    for (const p of batch) if (saveEntry(fromPost(p))) added++;
    batches++;
    const oldest = new Date(
      Math.min(...batch.map((p) => p.created_utc)) * 1000,
    );
    statusLine(
      `posts batch ${batches} — +${added} | oldest: ${oldest.toISOString().slice(0, 10)}`,
    );
    if (written >= TARGET) return;
    if (batches % 20 === 0) checkpoint();
  }
}

async function collectComments(): Promise<void> {
  let batches = 0;
  for await (const batch of walkBack<PullPushComment>("comment")) {
    let added = 0;
    for (const c of batch) if (saveEntry(fromComment(c))) added++;
    batches++;
    const oldest = new Date(
      Math.min(...batch.map((c) => c.created_utc)) * 1000,
    );
    statusLine(
      `comments batch ${batches} — +${added} | oldest: ${oldest.toISOString().slice(0, 10)}`,
    );
    if (written >= TARGET) break;
    if (batches % 20 === 0) checkpoint();
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main(): Promise<void> {
  console.log(
    `\nr/${SUBREDDIT} data collector (via PullPush.io — no credentials needed)`,
  );
  console.log(`Target  : ${TARGET.toLocaleString()} entries`);
  console.log(`Output  : ${DATA_DIR}`);
  console.log("─".repeat(60));

  // Quick connectivity check
  process.stdout.write("Checking PullPush.io… ");
  try {
    const test = await get<PullPushResponse<PullPushPost>>(
      `${BASE_URL}/submission/?subreddit=${SUBREDDIT}&limit=1`,
    );
    if (!Array.isArray(test.data))
      throw new Error("Unexpected response format");
    console.log(`OK (got ${test.data.length} test record)\n`);
  } catch (err) {
    console.error(`\n[error] Cannot reach PullPush.io: ${err}`);
    console.error("Check your internet connection and try again.");
    process.exit(1);
  }

  initOutput();

  if (written < TARGET) {
    process.stdout.write("Phase 1 — posts\n");
    await collectPosts();
    checkpoint();
    process.stdout.write(
      `\n  Phase 1 done — ${written.toLocaleString()} entries\n`,
    );
  }

  if (written < TARGET) {
    process.stdout.write("\nPhase 2 — comments\n");
    await collectComments();
    checkpoint();
    process.stdout.write(
      `\n  Phase 2 done — ${written.toLocaleString()} entries\n`,
    );
  }

  closeOutput();

  process.stdout.write("\n" + "─".repeat(60) + "\n");
  console.log(`Collection complete!`);
  console.log(`  Total entries : ${written.toLocaleString()}`);
  console.log(`  CSV           : ${CSV_PATH}`);
  console.log(`  JSONL         : ${JSONL_PATH}`);
}

main().catch((err) => {
  process.stdout.write("\n");
  console.error("Fatal error:", err);
  closeOutput();
  process.exit(1);
});
