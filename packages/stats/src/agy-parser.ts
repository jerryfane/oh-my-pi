/**
 * Ingestion of Google Antigravity CLI (`agy`) conversation transcripts into OMP stats.
 *
 * `agy` writes one compact JSONL transcript per conversation at:
 *   `<agyAppDataDir>/brain/<conversation-id>/.system_generated/logs/transcript.jsonl`
 * alongside a SQLite metadata index at:
 *   `<agyAppDataDir>/conversation_summaries.db`
 *
 * Both standalone `agy` sessions (main agents and `agy`-spawned subagents) and
 * OMP-delegated `agy -p` subagent runs write their authoritative per-LLM-turn
 * `PLANNER_RESPONSE` token usage to this transcript. Ingesting canonically from
 * `brain/` gives exact per-request rows, model IDs, tool calls, workspace folders,
 * and `main` vs `subagent` attribution while preventing any double-counting with
 * OMP's child `.jsonl` mirror transcripts.
 */

import { Database } from "bun:sqlite";
import type * as nodeFs from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import type { StopReason } from "@oh-my-pi/pi-ai";
import { getAgentDir, isEnoent, readLines } from "@oh-my-pi/pi-utils";
import type { ParseSessionResult, SessionParserState } from "./parser";
import { emptySpotlightRows } from "./spotlight-ingest";
import type {
	AgentType,
	MessageStatsInput,
	SessionEntry,
	SessionMessageEntry,
	ToolCallStats,
	UserMessageLink,
	UserMessageStats,
} from "./types";
import { computeUserMessageMetrics, judgeProse } from "./user-metrics";

/** Provider and API identifier used for all official `agy` CLI traffic in stats.db. */
export const AGY_CLI_PROVIDER = "agy-cli";
export const AGY_CLI_API = "agy-cli";

/**
 * Env var overriding the Antigravity CLI app-data root scanned by stats ingestion.
 * Set to `""` to disable `agy` transcript ingestion (used by isolated tests and
 * callers that only want OMP session files).
 */
export const STATS_AGY_DATA_DIR_ENV = "OMP_STATS_AGY_DATA_DIR";

const CHECKPOINT_BYTES = 256;
const TOOL_NAME_PATTERN = /^[\w.:-]+$/;
const SUMMARY_CACHE_TTL_MS = 15_000;
const AGY_ENTRY_ID_RE = /^agy:([^:]+):step:(\d+)$/;

/** Persisted per-file cursor state for incremental tailing of `agy` transcripts. */
export interface AgyParserState {
	agentType: AgentType;
	folder: string;
	sawFirstUserInput: boolean;
	pendingUserEntryId: string | null;
}

interface AgyConversationMeta {
	folder: string | null;
	isNestedSubagent: boolean;
}

interface SummaryCache {
	dbPath: string;
	fingerprint: string;
	loadedAt: number;
	byId: Map<string, AgyConversationMeta>;
}

let summaryCache: SummaryCache | null = null;

/**
 * Resolve the Antigravity CLI app-data directory (`~/.gemini/antigravity-cli`),
 * or `null` when explicitly disabled via `OMP_STATS_AGY_DATA_DIR=""` or when
 * running under an isolated OMP profile (`PI_CONFIG_DIR` / `setAgentDir` outside
 * `~/.omp/agent`) without an explicit `OMP_STATS_AGY_DATA_DIR` or `AGY_APP_DATA_DIR`.
 */
export function getAgyAppDataDir(): string | null {
	const explicit = process.env[STATS_AGY_DATA_DIR_ENV];
	if (explicit !== undefined) {
		const trimmed = explicit.trim();
		return trimmed.length > 0 ? path.resolve(trimmed) : null;
	}
	const agyEnv = process.env.AGY_APP_DATA_DIR?.trim();
	if (agyEnv) return path.resolve(agyEnv);
	const defaultAgentDir = path.resolve(os.homedir(), ".omp", "agent");
	if (path.resolve(getAgentDir()) !== defaultAgentDir) {
		return null;
	}
	return path.join(os.homedir(), ".gemini", "antigravity-cli");
}

/** Resolve `<agyAppDataDir>/brain`, or `null` when `agy` ingestion is disabled. */
export function getAgyBrainDir(): string | null {
	const root = getAgyAppDataDir();
	return root ? path.join(root, "brain") : null;
}

/**
 * Extract the `<conversation-id>` if `sessionPath` is an `agy` transcript at
 * `<agyBrainDir>/<conversation-id>/.system_generated/logs/transcript.jsonl`.
 */
export function extractAgyConversationId(sessionPath: string): string | null {
	const brainDir = getAgyBrainDir();
	if (!brainDir) return null;
	const rel = path.relative(brainDir, path.resolve(sessionPath));
	if (!rel || rel.startsWith("..") || path.isAbsolute(rel)) return null;
	const parts = rel.split(path.sep);
	if (
		parts.length === 4 &&
		parts[0].length > 0 &&
		parts[1] === ".system_generated" &&
		parts[2] === "logs" &&
		parts[3] === "transcript.jsonl"
	) {
		return parts[0];
	}
	return null;
}

/** Whether `sessionPath` points to an `agy` conversation transcript under `<agyBrainDir>`. */
export function isAgyTranscriptPath(sessionPath: string): boolean {
	return extractAgyConversationId(sessionPath) !== null;
}

/** List all `agy` conversation `transcript.jsonl` files under `<agyAppDataDir>/brain`. */
export async function listAgyTranscriptFiles(): Promise<string[]> {
	const brainDir = getAgyBrainDir();
	if (!brainDir) return [];
	let entries: nodeFs.Dirent[];
	try {
		entries = await fs.readdir(brainDir, { withFileTypes: true });
	} catch {
		return [];
	}
	const candidates = entries
		.filter(entry => entry.isDirectory() && !entry.name.startsWith("."))
		.map(entry => path.join(brainDir, entry.name, ".system_generated", "logs", "transcript.jsonl"));
	const checks = await Promise.all(
		candidates.map(async candidate => {
			try {
				const stat = await fs.stat(candidate);
				return stat.isFile() ? candidate : null;
			} catch {
				return null;
			}
		}),
	);
	return checks.filter((file): file is string => file !== null);
}

function parseWorkspaceUriToFolder(rawUris: string): string | null {
	if (!rawUris) return null;
	let parsed: unknown;
	try {
		parsed = JSON.parse(rawUris);
	} catch {
		return null;
	}
	if (!Array.isArray(parsed)) return null;
	for (const entry of parsed) {
		if (typeof entry !== "string") continue;
		const trimmed = entry.trim();
		if (!trimmed) continue;
		let folder: string | null = null;
		if (trimmed.startsWith("file://")) {
			try {
				folder = fileURLToPath(trimmed);
			} catch {
				folder = trimmed.slice("file://".length);
			}
		} else if (path.isAbsolute(trimmed)) {
			folder = trimmed;
		}
		if (folder) {
			return folder.length > 1 ? folder.replace(/\/+$/, "") : folder;
		}
	}
	return null;
}

async function summaryDbFingerprint(dbPath: string): Promise<string | null> {
	try {
		const mainStat = await fs.stat(dbPath);
		let walPart = "";
		try {
			const walStat = await fs.stat(`${dbPath}-wal`);
			walPart = `:${walStat.size}:${walStat.mtimeMs}`;
		} catch {
			// No active WAL file.
		}
		return `${mainStat.size}:${mainStat.mtimeMs}${walPart}`;
	} catch {
		return null;
	}
}

async function loadConversationMetaMap(): Promise<Map<string, AgyConversationMeta>> {
	const root = getAgyAppDataDir();
	if (!root) return new Map();
	const dbPath = path.join(root, "conversation_summaries.db");
	const fingerprint = await summaryDbFingerprint(dbPath);
	if (!fingerprint) return new Map();

	const now = Date.now();
	if (
		summaryCache &&
		summaryCache.dbPath === dbPath &&
		summaryCache.fingerprint === fingerprint &&
		now - summaryCache.loadedAt < SUMMARY_CACHE_TTL_MS
	) {
		return summaryCache.byId;
	}

	const byId = new Map<string, AgyConversationMeta>();
	let database: Database | null = null;
	try {
		database = new Database(dbPath, { readonly: true });
		database.run("PRAGMA busy_timeout = 1000");
		const rows = database
			.query<
				{
					conversation_id: string;
					workspace_uris: string;
					nesting_depth: number;
					parent_conversation_id: string;
				},
				[]
			>("SELECT conversation_id, workspace_uris, nesting_depth, parent_conversation_id FROM conversation_summaries")
			.all();
		for (const row of rows) {
			if (!row.conversation_id) continue;
			byId.set(row.conversation_id, {
				folder: parseWorkspaceUriToFolder(row.workspace_uris),
				isNestedSubagent:
					(typeof row.nesting_depth === "number" && row.nesting_depth > 0) ||
					(typeof row.parent_conversation_id === "string" && row.parent_conversation_id.trim().length > 0),
			});
		}
	} catch {
		// Summary DB may be absent, locked, or schema-different: fall back cleanly to transcript-only defaults.
		return summaryCache && summaryCache.dbPath === dbPath ? summaryCache.byId : byId;
	} finally {
		database?.close();
	}

	summaryCache = { dbPath, fingerprint, loadedAt: now, byId };
	return byId;
}

/** Look up the workspace folder for an `agy` transcript, falling back to `"agy"`. */
export async function resolveAgyConversationMeta(
	conversationId: string,
): Promise<{ folder: string; isNestedSubagent: boolean }> {
	const map = await loadConversationMetaMap();
	const found = map.get(conversationId);
	return {
		folder: found?.folder ?? "agy",
		isNestedSubagent: found?.isNestedSubagent ?? false,
	};
}

function stripUserRequestTags(content: string): string {
	let text = content.trim();
	if (text.startsWith("<USER_REQUEST>")) {
		text = text.slice("<USER_REQUEST>".length);
		const closeIdx = text.lastIndexOf("</USER_REQUEST>");
		if (closeIdx !== -1) text = text.slice(0, closeIdx);
	}
	return text.trim();
}

/**
 * Detect OMP's delegated subagent prompts inside `agy`'s `USER_INPUT` step so
 * OMP-spawned `agy -p` runs (including historical runs) classify as `subagent`.
 */
export function isDelegatedAgyPrompt(content: string): boolean {
	const stripped = stripUserRequestTags(content);
	if (stripped.startsWith('You are executing a delegated "')) return true;
	if (stripped.startsWith("Read the complete task instructions from ") && stripped.includes(".agy-prompt-")) {
		return true;
	}
	return false;
}

function isSyntheticAgyUserInput(content: string): boolean {
	const trimmed = content.trim();
	return (
		trimmed.startsWith("<CONTEXT_SUMMARY>") ||
		trimmed.startsWith("<SYSTEM_MESSAGE>") ||
		trimmed.startsWith("# Resuming from a compaction")
	);
}

function finiteNonNegInt(value: unknown): number {
	return typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.trunc(value) : 0;
}

function sanitizeToolName(name: string): string | null {
	const trimmed = name.trim();
	if (trimmed.length === 0) return null;
	if (TOOL_NAME_PATTERN.test(trimmed)) return trimmed;
	const candidate = trimmed.split(/[^\w.:-]/)[0] ?? "";
	return candidate.length > 0 ? candidate : null;
}

function parseIsoTimestampMs(raw: unknown): number {
	if (typeof raw !== "string" || !raw) return 0;
	const ms = Date.parse(raw);
	return Number.isFinite(ms) ? ms : 0;
}

async function readCheckpoint(handle: fs.FileHandle, end: number): Promise<Uint8Array> {
	const start = Math.max(0, end - CHECKPOINT_BYTES);
	const bytes = new Uint8Array(end - start);
	let read = 0;
	while (read < bytes.length) {
		const result = await handle.read(bytes, read, bytes.length - read, start + read);
		if (result.bytesRead === 0) break;
		read += result.bytesRead;
	}
	return bytes.subarray(0, read);
}

/**
 * Walk complete newline-terminated lines in `bytes`, calling `onRecord` for each
 * parsed JSON object. Returns the number of bytes consumed (never consuming a
 * trailing partial line without `\n`).
 */
function visitAgyJsonLines(bytes: Uint8Array, onRecord: (record: Record<string, unknown>) => void): number {
	const decoder = new TextDecoder();
	let pos = 0;
	while (pos < bytes.length) {
		const nl = bytes.indexOf(0x0a, pos);
		if (nl === -1) break;
		let lineEnd = nl;
		if (lineEnd > pos && bytes[lineEnd - 1] === 0x0d) lineEnd--;
		if (lineEnd > pos) {
			const line = decoder.decode(bytes.subarray(pos, lineEnd));
			try {
				const parsed: unknown = JSON.parse(line);
				if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
					onRecord(parsed as Record<string, unknown>);
				}
			} catch {
				// Skip malformed line while advancing past it.
			}
		}
		pos = nl + 1;
	}
	return pos;
}

/**
 * Incrementally parse an `agy` `transcript.jsonl` into the standard `ParseSessionResult`
 * shape consumed by `syncAllSessions` and `applySessionParseResults`.
 */
export async function parseAgyTranscriptFile(
	sessionPath: string,
	fromOffset = 0,
	state?: SessionParserState,
	replay = false,
): Promise<ParseSessionResult> {
	const conversationId = extractAgyConversationId(sessionPath) ?? path.basename(path.dirname(sessionPath));
	const meta = await resolveAgyConversationMeta(conversationId);

	const stats: MessageStatsInput[] = [];
	const userStats: UserMessageStats[] = [];
	const userLinks: UserMessageLink[] = [];
	const toolCalls: ToolCallStats[] = [];

	let info: nodeFs.Stats;
	let start = fromOffset;
	let reset = false;
	let checkpoint: string;
	let read = 0;
	let agyState: AgyParserState = {
		agentType: meta.isNestedSubagent ? "subagent" : "main",
		folder: meta.folder,
		sawFirstUserInput: false,
		pendingUserEntryId: null,
	};

	const visit = (record: Record<string, unknown>): void => {
		const type = record.type;
		const stepIndex =
			typeof record.step_index === "number" && Number.isFinite(record.step_index) ? record.step_index : -1;

		if (type === "USER_INPUT") {
			const content = typeof record.content === "string" ? record.content : "";
			if (!agyState.sawFirstUserInput) {
				agyState.sawFirstUserInput = true;
				if (isDelegatedAgyPrompt(content)) {
					agyState.agentType = "subagent";
				}
			}
			if (agyState.agentType === "main" && stepIndex >= 0 && !isSyntheticAgyUserInput(content)) {
				const text = stripUserRequestTags(content);
				if (text.length > 0) {
					const metrics = computeUserMessageMetrics(text);
					const prose = judgeProse(text);
					const entryId = `agy:${conversationId}:user:${stepIndex}`;
					userStats.push({
						sessionFile: sessionPath,
						entryId,
						folder: agyState.folder,
						timestamp: parseIsoTimestampMs(record.created_at),
						model: null,
						provider: null,
						chars: metrics.chars,
						words: metrics.words,
						yelling: metrics.yelling,
						profanity: metrics.profanity,
						anguish: metrics.anguish,
						negation: metrics.negation,
						repetition: metrics.repetition,
						blame: metrics.blame,
						prose,
						proseHash: prose ? Bun.hash(prose).toString(16) : "",
					});
					agyState.pendingUserEntryId = entryId;
				}
			}
			return;
		}

		if (type !== "PLANNER_RESPONSE") return;
		const status = typeof record.status === "string" ? record.status : "";
		if (status !== "DONE" && status !== "ERROR") return;
		const model = typeof record.model === "string" ? record.model.trim() : "";
		if (!model || stepIndex < 0) return;

		const input = finiteNonNegInt(record.input_tokens);
		const output = finiteNonNegInt(record.output_tokens);
		const cacheRead = finiteNonNegInt(record.cache_read_tokens);
		const totalTokens = input + output + cacheRead;
		if (totalTokens === 0 && status !== "ERROR") return;

		const timestamp = parseIsoTimestampMs(record.created_at);
		const rawToolCalls = Array.isArray(record.tool_calls) ? record.tool_calls : [];
		const stopReason: StopReason = status === "ERROR" ? "error" : rawToolCalls.length > 0 ? "toolUse" : "stop";
		const entryId = `agy:${conversationId}:step:${stepIndex}`;

		stats.push({
			sessionFile: sessionPath,
			entryId,
			folder: agyState.folder,
			model,
			provider: AGY_CLI_PROVIDER,
			api: AGY_CLI_API,
			timestamp,
			duration: null,
			ttft: null,
			stopReason,
			errorMessage: typeof record.error === "string" ? record.error : null,
			usage: {
				input,
				output,
				cacheRead,
				cacheWrite: 0,
				totalTokens,
				premiumRequests: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			agentType: agyState.agentType,
			serviceTier: null,
		});

		if (agyState.pendingUserEntryId) {
			userLinks.push({
				sessionFile: sessionPath,
				entryId: agyState.pendingUserEntryId,
				model,
				provider: AGY_CLI_PROVIDER,
			});
			agyState.pendingUserEntryId = null;
		}

		if (rawToolCalls.length > 0) {
			for (let idx = 0; idx < rawToolCalls.length; idx++) {
				const rawCall = rawToolCalls[idx];
				if (!rawCall || typeof rawCall !== "object" || Array.isArray(rawCall)) continue;
				const callObj = rawCall as { name?: unknown; args?: unknown };
				if (typeof callObj.name !== "string") continue;
				const toolName = sanitizeToolName(callObj.name);
				if (!toolName) continue;
				let argsChars = 0;
				try {
					argsChars = JSON.stringify(callObj.args ?? {}).length;
				} catch {
					// Ignore non-serializable tool args.
				}
				toolCalls.push({
					sessionFile: sessionPath,
					entryId,
					toolCallId: `${entryId}:tool:${idx}`,
					folder: agyState.folder,
					toolName,
					model,
					provider: AGY_CLI_PROVIDER,
					timestamp,
					agentType: agyState.agentType,
					callsInTurn: rawToolCalls.length,
					argsChars,
				});
			}
		}
	};

	try {
		const handle = await fs.open(sessionPath, "r");
		try {
			info = await handle.stat();
			const file = Bun.file(handle.fd);
			let resume = state?.version === 1 && state.offset === fromOffset && state.agy !== undefined;
			if (resume && state) {
				reset =
					state.ino !== info.ino ||
					state.birthtimeMs !== info.birthtimeMs ||
					info.size < state.size ||
					(info.size === state.size && info.mtimeMs !== state.mtimeMs);
				if (!reset) {
					const previous = await readCheckpoint(handle, fromOffset);
					reset = Bun.hash(previous).toString(16) !== state.checkpoint;
				}
				// If conversation_summaries.db resolved a real folder where an earlier pass fell back to "agy",
				// rebuild this transcript from byte 0 so stored folder attribution updates cleanly.
				if (!reset && state.agy && state.agy.folder === "agy" && meta.folder !== "agy") {
					reset = true;
				}
				if (!reset && state.agy && meta.isNestedSubagent && state.agy.agentType !== "subagent") {
					reset = true;
				}
				resume = !reset;
			}
			if (fromOffset > info.size) reset = true;
			if (replay) resume = false;
			start = reset || replay ? 0 : Math.max(0, fromOffset);
			const readStart = resume ? start : 0;
			const bytes = await file.slice(readStart, info.size).bytes();

			if (resume && state?.agy) {
				agyState = { ...state.agy };
			} else if (start > 0) {
				// Rebuild prefix classification state when starting mid-file without a saved cursor.
				visitAgyJsonLines(bytes.subarray(0, start), record => {
					if (record.type === "USER_INPUT" && !agyState.sawFirstUserInput) {
						agyState.sawFirstUserInput = true;
						if (typeof record.content === "string" && isDelegatedAgyPrompt(record.content)) {
							agyState.agentType = "subagent";
						}
					}
				});
			}

			read = visitAgyJsonLines(bytes.subarray(start - readStart), visit);
			const newOffset = start + read;
			const checkpointStart = Math.max(0, newOffset - CHECKPOINT_BYTES);
			const previous =
				checkpointStart >= readStart
					? bytes.subarray(checkpointStart - readStart, newOffset - readStart)
					: await readCheckpoint(handle, newOffset);
			checkpoint = Bun.hash(previous).toString(16);
		} finally {
			await handle.close();
		}
	} catch (err) {
		if (isEnoent(err)) {
			return {
				stats: [],
				userStats: [],
				userLinks: [],
				toolCalls: [],
				toolResults: [],
				toolApprovals: [],
				snapcompactCompactions: [],
				spotlight: emptySpotlightRows(),
				newOffset: fromOffset,
			};
		}
		throw err;
	}

	const newOffset = start + read;
	return {
		stats,
		userStats,
		userLinks,
		toolCalls,
		toolResults: [],
		toolApprovals: [],
		snapcompactCompactions: [],
		spotlight: emptySpotlightRows(),
		newOffset,
		reset,
		parserState: {
			version: 1,
			offset: newOffset,
			ino: info.ino,
			birthtimeMs: info.birthtimeMs,
			size: info.size,
			mtimeMs: info.mtimeMs,
			checkpoint,
			serviceTier: null,
			agy: agyState,
		},
	};
}

/**
 * Look up a single `PLANNER_RESPONSE` entry by synthetic `entryId` (`agy:<convId>:step:<N>`)
 * so clicking an `agy-cli` row in the dashboard's Recent Requests modal displays its payload.
 */
export async function getAgySessionEntry(sessionPath: string, entryId: string): Promise<SessionEntry | null> {
	const match = AGY_ENTRY_ID_RE.exec(entryId);
	if (!match) return null;
	const targetStep = Number(match[2]);
	if (!Number.isFinite(targetStep)) return null;

	const decoder = new TextDecoder();
	try {
		for await (const rawLine of readLines(Bun.file(sessionPath).stream())) {
			if (rawLine.length === 0) continue;
			let parsed: unknown;
			try {
				parsed = JSON.parse(decoder.decode(rawLine));
			} catch {
				continue;
			}
			if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) continue;
			const record = parsed as Record<string, unknown>;
			if (record.type !== "PLANNER_RESPONSE" || record.step_index !== targetStep) continue;

			const content: Array<Record<string, unknown>> = [];
			if (typeof record.thinking === "string" && record.thinking.length > 0) {
				content.push({ type: "thinking", thinking: record.thinking });
			}
			if (typeof record.content === "string" && record.content.length > 0) {
				content.push({ type: "text", text: record.content });
			}
			if (Array.isArray(record.tool_calls)) {
				for (let idx = 0; idx < record.tool_calls.length; idx++) {
					const tc = record.tool_calls[idx] as { name?: unknown; args?: unknown } | null;
					if (tc && typeof tc.name === "string") {
						content.push({
							type: "toolCall",
							id: `${entryId}:tool:${idx}`,
							name: tc.name,
							arguments: tc.args && typeof tc.args === "object" ? tc.args : {},
						});
					}
				}
			}
			const input = finiteNonNegInt(record.input_tokens);
			const output = finiteNonNegInt(record.output_tokens);
			const cacheRead = finiteNonNegInt(record.cache_read_tokens);
			const tsIso = typeof record.created_at === "string" ? record.created_at : new Date(0).toISOString();
			const entry: SessionMessageEntry = {
				type: "message",
				id: entryId,
				parentId: null,
				timestamp: tsIso,
				message: {
					role: "assistant",
					content: content as never,
					api: AGY_CLI_API as never,
					provider: AGY_CLI_PROVIDER as never,
					model: typeof record.model === "string" ? record.model : "unknown",
					usage: {
						input,
						output,
						cacheRead,
						cacheWrite: 0,
						totalTokens: input + output + cacheRead,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
					stopReason:
						record.status === "ERROR"
							? "error"
							: Array.isArray(record.tool_calls) && record.tool_calls.length > 0
								? "toolUse"
								: "stop",
					timestamp: parseIsoTimestampMs(record.created_at),
				},
			};
			return entry;
		}
	} catch (err) {
		if (isEnoent(err)) return null;
		throw err;
	}
	return null;
}

/** Clear the in-memory `conversation_summaries.db` cache (for tests). */
export function resetAgySummaryCache(): void {
	summaryCache = null;
}
