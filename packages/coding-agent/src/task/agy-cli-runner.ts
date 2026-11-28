/**
 * Runs delegated OMP subagents via the official `agy --input-format stream-json --output-format stream-json` CLI.
 *
 * Unlike OMP's ACP provider (which proxies OMP's own prompt/tools over `agy acp`), this runner
 * delegates the whole subagent assignment to the official `agy` CLI so `agy` runs normally with
 * its own prompt and tools while streaming live progress, tool calls, usage, artifacts,
 * mid-job queued follow-ups, immediate mid-turn steers, post-job revival, and structured results
 * straight into OMP's `Subagents` (`Alt+A`) HUD, `IrcBus`, RPC controls, `wait`, and `proc://`.
 */

import * as crypto from "node:crypto";
import * as fsSync from "node:fs";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { AgentMessage, ThinkingLevel } from "@oh-my-pi/pi-agent-core";
import type { ImageContent, Model } from "@oh-my-pi/pi-ai";
import { $env, $which, isRecord, logger, ptree } from "@oh-my-pi/pi-utils";
import type { EffectiveExtensionRoots } from "../capability/types";
import { cfgOffline } from "../config/auxiliary-model-policy";
import { Settings } from "../config/settings";
import { IrcBus } from "../irc/bus";
import type { IrcMessage } from "../irc/irc-types";
import { AgentLifecycleManager, type AgentReviver } from "../registry/agent-lifecycle";
import { type AgentRef, AgentRegistry, MAIN_AGENT_ID } from "../registry/agent-registry";
import type { AgentSessionEvent, AgentSessionEventListener } from "../session/agent-session-events";
import type {
	AgentModelChange,
	AgentModelSelectionState,
	AgentSessionAdvisorOverview,
	AgentSessionHandle,
	AgentSessionLoopHandle,
	AgentSessionManagerHandle,
} from "../session/agent-session-handle";
import type {
	AgentSessionDisposeOptions,
	PromptOptions,
	SessionStats,
	SteerOptions,
} from "../session/agent-session-types";
import type { CustomMessage } from "../session/messages";
import type { ServingModel } from "../session/retry-fallback-chains";
import { CURRENT_SESSION_VERSION, type UsageStatistics } from "../session/session-entries";
import { cfgSessionMaxCost } from "../session/settings";
import { isMainThread } from "../thread/identity";
import { registerLocalBridgedHandle } from "../thread/session-directory";
import { buildOutputValidator, summarizeValidationFailure } from "../tools/output-schema-validator";
import { truncateTail } from "../tools/streaming-output";
import type { EventBus } from "../utils/event-bus";
import { previewLine, shortenToolArgumentPaths, TRUNCATE_LENGTHS } from "../utils/render-utils";
import type { ExecutorOptions, FollowUpTurnOptions } from "./executor";
import { persistSubagentOutputArtifacts } from "./output-artifacts";
import { isReadOnlyAgent } from "./read-only-policy";
import { cfgTaskAgentIdleTtlMs, cfgTaskAgyBinary, cfgTaskAgySubagents, cfgTaskMaxRuntimeMs } from "./settings";
import type { AgentProgress, SingleResult, StructuredSubagentOutput } from "./task-details";
import {
	type AgentDefinition,
	MAX_OUTPUT_BYTES,
	MAX_OUTPUT_LINES,
	TASK_SUBAGENT_LIFECYCLE_CHANNEL,
	TASK_SUBAGENT_PROGRESS_CHANNEL,
} from "./types";
import type { WorkPoolYieldItem } from "./workpool-yield";

const PROGRESS_COALESCE_MS = 150;
const POST_EXIT_DRAIN_MS = 250;
const RECENT_TOOLS_LIMIT = 10;
const RECENT_OUTPUT_LINES = 8;
const RECENT_OUTPUT_TAIL_BYTES = 8 * 1024;
const TRANSCRIPT_MAX_TOOL_ARGS_BYTES = 16 * 1024;
const TRANSCRIPT_MAX_TOOL_OUTPUT_BYTES = 64 * 1024;
const TRANSCRIPT_MAX_TOOL_OUTPUT_LINES = 1000;
const SAFE_TRANSCRIPT_ID_RE = /^[A-Za-z0-9_.-]+$/;
const DISABLED_VALUES = new Set(["off", "false", "none", "disabled", "0"]);
const DEFAULT_MODEL_VALUES = new Set(["default", "auto", "true", "on", "1"]);

export interface AgySubagentConfig {
	binary: string;
	/** Explicit `agy --model` argument, or `undefined` to let `agy` use its default model. */
	model?: string;
}

interface AgyUsage {
	input_tokens?: number;
	output_tokens?: number;
	thinking_tokens?: number;
	cache_read_tokens?: number;
	total_tokens?: number;
}

interface AgyStreamInit {
	event: "init";
	conversation_id?: string;
	init?: {
		model?: string;
		cwd?: string;
		permission_mode?: string;
	};
}

interface AgyStreamStepUpdate {
	event: "step_update";
	step_update?: {
		conversation_id?: string;
		step_index?: number;
		state?: string;
		step_type?: string;
		tool_name?: string;
		tool_info?: {
			name?: string;
			parameters?: Record<string, unknown>;
			output?: string;
		};
		text_delta?: string;
		duration_seconds?: number;
		usage?: AgyUsage;
	};
}

interface AgyStreamResult {
	event: "result";
	malformedReason?: string;
	result?: {
		conversation_id?: string;
		status?: string;
		response?: string;
		error?: string;
		duration_seconds?: number;
		num_turns?: number;
		structured_output?: unknown;
		usage?: AgyUsage;
		finished_reason?: string;
	};
}

export type AgyStreamEvent = AgyStreamInit | AgyStreamStepUpdate | AgyStreamResult;

function optionalString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function optionalFiniteNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function parseAgyUsage(raw: unknown): AgyUsage | undefined {
	if (!isRecord(raw)) return undefined;
	const usage: AgyUsage = {};
	const input = optionalFiniteNumber(raw.input_tokens);
	const output = optionalFiniteNumber(raw.output_tokens);
	const thinking = optionalFiniteNumber(raw.thinking_tokens);
	const cacheRead = optionalFiniteNumber(raw.cache_read_tokens);
	const total = optionalFiniteNumber(raw.total_tokens);
	if (input !== undefined && input >= 0) usage.input_tokens = input;
	if (output !== undefined && output >= 0) usage.output_tokens = output;
	if (thinking !== undefined && thinking >= 0) usage.thinking_tokens = thinking;
	if (cacheRead !== undefined && cacheRead >= 0) usage.cache_read_tokens = cacheRead;
	if (total !== undefined && total >= 0) usage.total_tokens = total;
	return usage;
}

export function parseAgyStreamLine(line: string): AgyStreamEvent | undefined {
	const trimmed = line.trim();
	if (!trimmed.startsWith("{")) return undefined;
	try {
		const parsed: unknown = JSON.parse(trimmed);
		if (!isRecord(parsed) || typeof parsed.event !== "string") return undefined;
		if (parsed.event === "init") {
			const rawInit = isRecord(parsed.init) ? parsed.init : undefined;
			return {
				event: "init",
				conversation_id: optionalString(parsed.conversation_id),
				...(rawInit
					? {
							init: {
								model: optionalString(rawInit.model),
								cwd: optionalString(rawInit.cwd),
								permission_mode: optionalString(rawInit.permission_mode),
							},
						}
					: {}),
			};
		}
		if (parsed.event === "step_update") {
			const rawSu = isRecord(parsed.step_update) ? parsed.step_update : undefined;
			if (!rawSu) return { event: "step_update" };
			const rawToolInfo = isRecord(rawSu.tool_info) ? rawSu.tool_info : undefined;
			return {
				event: "step_update",
				step_update: {
					conversation_id: optionalString(rawSu.conversation_id),
					step_index: optionalFiniteNumber(rawSu.step_index),
					state: optionalString(rawSu.state),
					step_type: optionalString(rawSu.step_type),
					tool_name: optionalString(rawSu.tool_name),
					...(rawToolInfo
						? {
								tool_info: {
									name: optionalString(rawToolInfo.name),
									parameters: isRecord(rawToolInfo.parameters) ? rawToolInfo.parameters : undefined,
									output: optionalString(rawToolInfo.output),
								},
							}
						: {}),
					text_delta: optionalString(rawSu.text_delta),
					duration_seconds: optionalFiniteNumber(rawSu.duration_seconds),
					usage: parseAgyUsage(rawSu.usage),
				},
			};
		}
		if (parsed.event === "result") {
			if (!isRecord(parsed.result)) {
				return {
					event: "result",
					malformedReason: "agy emitted a malformed terminal result frame: missing result object",
				};
			}
			const rawRes = parsed.result;
			let malformedReason: string | undefined;
			if (rawRes.status !== undefined && typeof rawRes.status !== "string") {
				malformedReason = "agy emitted a malformed terminal result frame: non-string status";
			} else if (rawRes.response !== undefined && typeof rawRes.response !== "string") {
				malformedReason = "agy emitted a malformed terminal result frame: non-string response";
			} else if (rawRes.error !== undefined && typeof rawRes.error !== "string") {
				malformedReason = "agy emitted a malformed terminal result frame: non-string error";
			}
			const numTurns = optionalFiniteNumber(rawRes.num_turns);
			return {
				event: "result",
				...(malformedReason ? { malformedReason } : {}),
				result: {
					conversation_id: optionalString(rawRes.conversation_id),
					status: optionalString(rawRes.status),
					response: optionalString(rawRes.response),
					error: optionalString(rawRes.error),
					duration_seconds: optionalFiniteNumber(rawRes.duration_seconds),
					num_turns: numTurns !== undefined && numTurns >= 0 ? numTurns : undefined,
					structured_output: rawRes.structured_output,
					usage: parseAgyUsage(rawRes.usage),
					finished_reason: optionalString(rawRes.finished_reason),
				},
			};
		}
		return undefined;
	} catch {
		return undefined;
	}
}

function isExecutableFile(candidate: string): boolean {
	try {
		fsSync.accessSync(candidate, fsSync.constants.X_OK);
		return fsSync.statSync(candidate).isFile();
	} catch {
		return false;
	}
}

function formatErrorMessage(err: unknown): string {
	try {
		if (err instanceof Error) return String(err.message);
		return String(err);
	} catch {
		return "unknown error";
	}
}

function invokeObserverSafely(label: string, context: Record<string, unknown>, fn: () => unknown): void {
	let result: unknown;
	try {
		result = fn();
	} catch (err) {
		logger.warn(label, { ...context, error: formatErrorMessage(err) });
		return;
	}
	if (result !== null && (typeof result === "object" || typeof result === "function")) {
		// Assimilate any Promise/PromiseLike (including virtual-then proxies and promises with
		// shadowed `.catch`) via native `await` in a detached async wrapper without delaying callers.
		void (async () => {
			try {
				await result;
			} catch (err) {
				logger.warn(label, { ...context, error: formatErrorMessage(err) });
			}
		})();
	}
}

function emitSubagentFrameSafely(
	eventBus: EventBus | undefined,
	subagentEventBus: EventBus | undefined,
	channel: string,
	payload: unknown,
	context: Record<string, unknown>,
): void {
	if (eventBus) {
		invokeObserverSafely("agy eventBus emit failed", { ...context, channel, bus: "session" }, () =>
			eventBus.emit(channel, payload),
		);
	}
	if (subagentEventBus && subagentEventBus !== eventBus) {
		invokeObserverSafely("agy subagentEventBus emit failed", { ...context, channel, bus: "observability" }, () =>
			subagentEventBus.emit(channel, payload),
		);
	}
}

function resolveAgyBinaryPath(settings: Settings): string | undefined {
	const configured = cfgTaskAgyBinary.get(settings)?.trim() || $env.OMP_AGY_SUBAGENT_BIN?.trim();
	if (configured) {
		if (configured.includes("/") || configured.includes("\\")) {
			// Canonicalize via OS realpath(3) (`realpathSync.native`, avoiding Bun's lexical `realpathSync`
			// normalization of `link/..`) so launching from another task cwd/worktree executes the exact target binary.
			try {
				const resolved = fsSync.realpathSync.native(configured);
				return isExecutableFile(resolved) ? resolved : undefined;
			} catch {
				return undefined;
			}
		}
		return $which(configured) ?? undefined;
	}
	return $which("agy") ?? undefined;
}

function normalizeSelector(raw: unknown): string | undefined {
	if (typeof raw === "boolean") return raw ? "default" : "off";
	if (typeof raw === "string") {
		const trimmed = raw.trim();
		return trimmed.length > 0 ? trimmed : undefined;
	}
	return undefined;
}

/**
 * Decide whether a subagent spawn should execute via the official `agy -p` CLI.
 * Returns `undefined` to fall back cleanly to OMP's native in-process subagent executor.
 */
export function resolveAgySubagentConfig(options: ExecutorOptions): AgySubagentConfig | undefined {
	const settings = options.settings;
	if (!settings) return undefined;
	// Worker threads bridge registry sessions through `AgentSession`-only codecs; run `agy` on the main thread.
	if (!isMainThread()) return undefined;

	// Enforce OMP security, offline, spend-cap, transcript-epoch, and resumable-lifecycle boundaries natively.
	if (cfgOffline.get(settings)) return undefined;
	if (options.costGate?.capped === true || cfgSessionMaxCost.get(settings) !== undefined) return undefined;
	if (isReadOnlyAgent(options.agent)) return undefined;
	if (options.perCallModelGrants !== undefined) return undefined;
	if (options.restrictToolNames === true) return undefined;
	if (options.customTools !== undefined && options.customTools.length > 0) return undefined;
	if (options.workPoolYieldItems !== undefined && options.workPoolYieldItems.length > 0) return undefined;
	if (options.topicBoard !== undefined) return undefined;
	if (options.launchPreamble !== undefined) return undefined;
	if (options.launchSignal !== undefined) return undefined;
	if (options.onFirstTurn !== undefined) return undefined;

	const rawMapping: unknown = cfgTaskAgySubagents.get(settings);
	if (!isRecord(rawMapping)) return undefined;
	const agentName = options.agent.name;
	const rawEntry = Object.hasOwn(rawMapping, agentName)
		? rawMapping[agentName]
		: Object.hasOwn(rawMapping, "*")
			? rawMapping["*"]
			: undefined;
	const normalized = normalizeSelector(rawEntry);
	if (!normalized) return undefined;

	const lower = normalized.toLowerCase();
	if (DISABLED_VALUES.has(lower)) return undefined;

	const binary = resolveAgyBinaryPath(settings);
	if (!binary) {
		logger.warn("task.agySubagents configured, but `agy` binary was not found; falling back to native OMP subagent", {
			agent: agentName,
		});
		return undefined;
	}

	const model = DEFAULT_MODEL_VALUES.has(lower) ? undefined : normalized;
	return { binary, model };
}

export function buildAgyPrompt(options: ExecutorOptions, stepPrompt: string, includeHeader: boolean): string {
	const sections: string[] = [];
	if (includeHeader) {
		sections.push(
			[
				`You are executing a delegated "${options.agent.name}" subagent task inside a larger workflow.`,
				"Return a concise, accurate final answer directly addressing the assignment (no meta-commentary about being a subagent).",
			].join("\n"),
		);

		const agentSystemPrompt = options.agent.systemPrompt?.trim();
		if (agentSystemPrompt) {
			sections.push(`<agent_role>\n${agentSystemPrompt}\n</agent_role>`);
		}
		if (options.context?.trim()) {
			sections.push(`<shared_context>\n${options.context.trim()}\n</shared_context>`);
		}
		if (options.planReference?.content.trim()) {
			sections.push(
				`<active_plan path="${options.planReference.path}">\n${options.planReference.content.trim()}\n</active_plan>`,
			);
		}
	}
	sections.push(stepPrompt);
	return sections.join("\n\n");
}

function summarizeToolArgs(
	parameters: Record<string, unknown> | undefined,
	cwd: string,
): {
	args: string;
	argsKey?: string;
} {
	if (!parameters) return { args: "" };
	const priorityKeys = [
		"Path",
		"path",
		"CommandLine",
		"command",
		"Query",
		"query",
		"Pattern",
		"pattern",
		"TargetFile",
		"SearchPath",
		"DirectoryPath",
		"Url",
		"url",
	];
	for (const key of priorityKeys) {
		const val = parameters[key];
		if (typeof val === "string" && val.trim()) {
			const shortened = shortenToolArgumentPaths(val.trim(), cwd);
			return { args: previewLine(shortened, TRUNCATE_LENGTHS.SHORT), argsKey: key };
		}
	}
	for (const [key, val] of Object.entries(parameters)) {
		if (typeof val === "string" && val.trim()) {
			const shortened = shortenToolArgumentPaths(val.trim(), cwd);
			return { args: previewLine(shortened, TRUNCATE_LENGTHS.SHORT), argsKey: key };
		}
	}
	return { args: "" };
}

function formatResolvedAgyModel(model: string | undefined): string {
	return model ? `agy-cli/${model}` : "agy-cli/default";
}

function bareAgyModelName(modelDisplay: string): string {
	return modelDisplay.startsWith("agy-cli/") ? modelDisplay.slice("agy-cli/".length) : modelDisplay;
}

function abortReasonOf(signal: AbortSignal | undefined, fallback: string): string {
	if (!signal?.aborted) return fallback;
	try {
		const reason = signal.reason;
		if (typeof reason === "string" && reason.trim()) return reason.trim();
		if (reason instanceof Error) {
			const message = String(reason.message).trim();
			if (message) return message;
		}
	} catch {
		// Fall back if signal.reason or its getters throw.
	}
	return fallback;
}

const STEER_INTERRUPT_GRACE_MS = 500;

function formatAgyStreamUserInput(content: string): string {
	return `${JSON.stringify({ event: "user", message: { role: "user", content } })}\n`;
}

function buildStreamArgs(params: {
	binary: string;
	model?: string;
	conversationId?: string;
	additionalDirectories?: string[];
	jsonSchema?: string;
}): string[] {
	const args = [
		params.binary,
		"--input-format",
		"stream-json",
		"--output-format",
		"stream-json",
		"--dangerously-skip-permissions",
	];
	if (params.model) {
		args.push("--model", params.model);
	}
	if (params.conversationId) {
		args.push("--conversation", params.conversationId);
	}
	if (params.additionalDirectories) {
		for (const dir of params.additionalDirectories) {
			if (dir.trim()) args.push("--add-dir", dir);
		}
	}
	if (params.jsonSchema) {
		args.push("--json-schema", params.jsonSchema);
	}
	return args;
}

function buildJsonSchemaPromptInstruction(jsonSchema?: string): string {
	if (jsonSchema) {
		return [
			"<output_format>",
			"Your final response MUST be a single raw valid JSON value (no markdown code fences and no surrounding prose) conforming to this JSON Schema:",
			jsonSchema,
			"</output_format>",
		].join("\n");
	}
	return [
		"<output_format>",
		"Your final response MUST be a single raw valid JSON value (no markdown code fences and no surrounding prose).",
		"</output_format>",
	].join("\n");
}

interface StepUsageTotals {
	requests: number;
	inputTokens: number;
	outputTokens: number;
	cacheReadTokens: number;
	lastContextTokens?: number;
}

interface UnconfirmedFollowUp {
	text: string;
	alreadyPublishedInTranscript: boolean;
}

interface StepRunOutcome {
	exitCode: number;
	conversationId?: string;
	model?: string;
	response: string;
	responseIsAuthoritative: boolean;
	structuredOutput?: unknown;
	usage: StepUsageTotals;
	durationMs: number;
	error?: string;
	stderr: string;
	aborted: boolean;
	timedOut: boolean;
	abortReason?: string;
	interruptedForSteer?: boolean;
	/** Follow-up prompts accepted for this step whose turn was not confirmed by a `result` frame before exit. */
	unconfirmedFollowUps?: UnconfirmedFollowUp[];
	/** True when the last completed `result` frame was already recorded in the transcript via an intermediate transition. */
	lastResultAlreadyPublished?: boolean;
}

interface ActiveAgyProcessControl {
	/** Write a queued next-turn NDJSON user message to `agy`'s live stdin. Returns false if stdin is closed. */
	writeFollowUp(content: string): boolean;
	/** Interrupt the active turn (`SIGINT` + bounded tree cleanup) so the runner immediately resumes via `--conversation`. */
	interruptForSteer(): boolean;
}

async function runSingleAgyStep(params: {
	binary: string;
	cwd: string;
	model?: string;
	conversationId?: string;
	additionalDirectories?: string[];
	jsonSchema?: string;
	promptText: string;
	id: string;
	stepIndex: number;
	abortSignal?: AbortSignal;
	deadlineMs?: number;
	maxRuntimeMs: number;
	onEvent: (event: AgyStreamEvent) => void;
	onProcessAttached?: (control: ActiveAgyProcessControl) => void;
	onProcessDetached?: () => void;
	onConversationId?: (conversationId: string) => void;
	onIntermediateTurnComplete?: (info: {
		prevResult: AgyStreamResult["result"];
		prevUsage: StepUsageTotals;
		nextPrompt: string;
	}) => void;
	drainQueuedFollowUps?: () => string[];
	hasPendingSteer?: () => boolean;
}): Promise<StepRunOutcome> {
	const stepStartTime = Date.now();
	let aborted = false;
	let timedOut = false;
	let abortReason: string | undefined;
	let steerInterrupted = false;
	let conversationId = params.conversationId;
	let model = params.model;
	let streamedText = "";
	let resultEvent: AgyStreamResult["result"] | undefined;
	let resultMalformedReason: string | undefined;
	let stderrText = "";
	let stepRequests = 0;
	let stepInput = 0;
	let stepOutput = 0;
	let stepCacheRead = 0;
	let stepLastContext: number | undefined;
	let cp: ptree.ChildProcess<"pipe"> | undefined;
	let stdinClosed = false;
	let pendingTurns = 0;
	let resultsReceived = 0;
	let lastResultPublished = false;
	let activeTurnHasResult = false;
	let successorTransitionObserved = false;
	const writtenFollowUps: Array<{ text: string; publishedInTranscript: boolean }> = [];
	const unwrittenFollowUps: string[] = [];

	const currentUsage = (includeSingleRequestFallback = false): StepUsageTotals => {
		let lastContextTokens = stepLastContext;
		if (
			includeSingleRequestFallback &&
			lastContextTokens === undefined &&
			activeTurnHasResult &&
			!successorTransitionObserved &&
			resultsReceived === 1 &&
			stepRequests === 1 &&
			resultEvent?.usage
		) {
			const ctx = (resultEvent.usage.input_tokens ?? 0) + (resultEvent.usage.cache_read_tokens ?? 0);
			if (ctx > 0) lastContextTokens = ctx;
		}
		return {
			requests: stepRequests,
			inputTokens: stepInput,
			outputTokens: stepOutput,
			cacheReadTokens: stepCacheRead,
			lastContextTokens,
		};
	};

	if (params.abortSignal?.aborted) {
		abortReason = abortReasonOf(params.abortSignal, "Cancelled");
		return {
			exitCode: 1,
			response: "",
			responseIsAuthoritative: false,
			usage: currentUsage(),
			durationMs: 0,
			error: abortReason,
			stderr: abortReason,
			aborted: true,
			timedOut: false,
			abortReason,
		};
	}

	if (params.deadlineMs !== undefined && Date.now() >= params.deadlineMs) {
		abortReason = `Subagent exceeded task.maxRuntimeMs (${params.maxRuntimeMs}ms)`;
		return {
			exitCode: 1,
			response: "",
			responseIsAuthoritative: false,
			usage: currentUsage(),
			durationMs: 0,
			error: abortReason,
			stderr: abortReason,
			aborted: true,
			timedOut: true,
			abortReason,
		};
	}

	const closeStdinSafely = () => {
		if (stdinClosed || !cp) return;
		stdinClosed = true;
		try {
			const endResult = cp.stdin.end();
			if (endResult instanceof Promise) {
				void endResult.catch(() => {});
			}
		} catch {
			// Stdin may already be closed if process exited early.
		}
	};

	const writePromptToStdin = (content: string, isFollowUp: boolean): boolean => {
		if (stdinClosed || !cp || steerInterrupted || params.abortSignal?.aborted) return false;
		try {
			cp.stdin.write(formatAgyStreamUserInput(content));
			const flushResult = cp.stdin.flush();
			if (flushResult instanceof Promise) {
				void flushResult.catch(() => {
					stdinClosed = true;
				});
			}
			pendingTurns += 1;
			if (isFollowUp) {
				writtenFollowUps.push({ text: content, publishedInTranscript: false });
			}
			return true;
		} catch {
			stdinClosed = true;
			return false;
		}
	};

	const flushDrainedFollowUps = () => {
		for (const queued of params.drainQueuedFollowUps?.() ?? []) {
			if (!writePromptToStdin(queued, true)) {
				unwrittenFollowUps.push(queued);
			}
		}
	};

	const interruptForSteer = (): boolean => {
		if (!cp || steerInterrupted || params.abortSignal?.aborted) return false;
		steerInterrupted = true;
		closeStdinSafely();
		cp.kill(new ptree.AbortError("steer", ""), STEER_INTERRUPT_GRACE_MS, "SIGINT");
		return true;
	};

	const noteConversationId = (rawCid: string | undefined) => {
		const trimmed = rawCid?.trim();
		if (!trimmed) return;
		const isNew = trimmed !== conversationId;
		conversationId = trimmed;
		if (isNew) {
			params.onConversationId?.(trimmed);
		}
		if (params.hasPendingSteer?.()) {
			interruptForSteer();
		}
	};

	const collectUnconfirmedFollowUps = (): UnconfirmedFollowUp[] => {
		const confirmedCount = Math.max(0, resultsReceived - 1);
		return [
			...writtenFollowUps.slice(confirmedCount).map(item => ({
				text: item.text,
				alreadyPublishedInTranscript: item.publishedInTranscript,
			})),
			...unwrittenFollowUps.map(text => ({ text, alreadyPublishedInTranscript: false })),
		];
	};

	try {
		let cmd = buildStreamArgs({
			binary: params.binary,
			model: params.model,
			conversationId: params.conversationId,
			additionalDirectories: params.additionalDirectories,
			jsonSchema: params.jsonSchema,
		});
		let initialPrompt = params.promptText;
		// Keep `--json-schema` on argv when it fits; if the schema itself overflows supervisor launch limits,
		// fold it into the stdin prompt and still validate the response locally.
		if (params.jsonSchema !== undefined && !ptree.fitsInlineSpawn(cmd, { subreaper: true })) {
			initialPrompt = `${params.promptText}\n\n${buildJsonSchemaPromptInstruction(params.jsonSchema)}`;
			cmd = buildStreamArgs({
				binary: params.binary,
				model: params.model,
				conversationId: params.conversationId,
				additionalDirectories: params.additionalDirectories,
			});
		}

		const remainingMs = params.deadlineMs !== undefined ? params.deadlineMs - Date.now() : undefined;
		if (remainingMs !== undefined && remainingMs <= 0) {
			abortReason = `Subagent exceeded task.maxRuntimeMs (${params.maxRuntimeMs}ms)`;
			return {
				exitCode: 1,
				response: "",
				responseIsAuthoritative: false,
				usage: currentUsage(),
				durationMs: Math.max(0, Date.now() - stepStartTime),
				error: abortReason,
				stderr: abortReason,
				aborted: true,
				timedOut: true,
				abortReason,
			};
		}

		// Supervise the entire process tree (detached + Linux subreaper) with an open NDJSON stdin pipe so
		// mid-job queued messages run seamlessly as follow-up turns in the same live `agy` process.
		cp = ptree.spawn(cmd, {
			cwd: params.cwd,
			stdin: "pipe",
			detached: true,
			subreaper: true,
			descendantDrainMs: POST_EXIT_DRAIN_MS,
			...(params.abortSignal ? { signal: params.abortSignal } : {}),
			...(remainingMs !== undefined ? { timeout: remainingMs } : {}),
			env: { ...Bun.env, AGY_CLI_DISABLE_AUTO_UPDATE: "1" },
		});

		params.onProcessAttached?.({
			writeFollowUp: content => writePromptToStdin(content, true),
			interruptForSteer,
		});

		writePromptToStdin(initialPrompt, false);
		// Flush any queued messages that arrived while spawning before the process attached.
		flushDrainedFollowUps();

		const decoder = new TextDecoder();
		let pending = "";
		let pendingIntermediateTransition:
			| {
					prevResult: AgyStreamResult["result"];
					prevUsage: StepUsageTotals;
					followUp: { text: string; publishedInTranscript: boolean };
			  }
			| undefined;
		const reconcileResultUsage = (res: AgyStreamResult["result"] | undefined) => {
			if (!res) return;
			if (res.num_turns !== undefined) {
				stepRequests = res.num_turns;
			}
			const resUsage = res.usage;
			if (resUsage) {
				if (resUsage.input_tokens !== undefined) {
					stepInput = resUsage.input_tokens;
				}
				if (resUsage.output_tokens !== undefined) {
					stepOutput = resUsage.output_tokens;
				}
				if (resUsage.cache_read_tokens !== undefined) {
					stepCacheRead = resUsage.cache_read_tokens;
				}
			}
		};
		const flushPendingIntermediateTransition = () => {
			if (!pendingIntermediateTransition) return;
			const transition = pendingIntermediateTransition;
			pendingIntermediateTransition = undefined;
			transition.followUp.publishedInTranscript = true;
			lastResultPublished = true;
			activeTurnHasResult = false;
			successorTransitionObserved = true;
			stepLastContext = undefined;
			params.onIntermediateTurnComplete?.({
				prevResult: transition.prevResult,
				prevUsage: transition.prevUsage,
				nextPrompt: transition.followUp.text,
			});
		};
		const handleLine = (rawLine: string) => {
			const parsed = parseAgyStreamLine(rawLine);
			if (!parsed) return;
			if (parsed.event === "init") {
				noteConversationId(parsed.conversation_id);
				if (parsed.init?.model) model = parsed.init.model;
			} else if (parsed.event === "step_update") {
				flushPendingIntermediateTransition();
				const su = parsed.step_update;
				noteConversationId(su?.conversation_id);
				if (su?.step_type === "agent_response") {
					if (su.text_delta) streamedText += su.text_delta;
					if (su.state === "DONE") {
						stepRequests += 1;
						if (su.usage) {
							const inp = su.usage.input_tokens ?? 0;
							const out = su.usage.output_tokens ?? 0;
							const cacheRead = su.usage.cache_read_tokens ?? 0;
							stepInput += inp;
							stepOutput += out;
							stepCacheRead += cacheRead;
							const ctx = inp + cacheRead;
							if (ctx > 0) stepLastContext = ctx;
						}
					}
				}
			} else if (parsed.event === "result") {
				flushPendingIntermediateTransition();
				resultEvent = parsed.result;
				resultMalformedReason = parsed.malformedReason;
				reconcileResultUsage(parsed.result);
				activeTurnHasResult = true;
				lastResultPublished = false;
				resultsReceived += 1;
				noteConversationId(parsed.result?.conversation_id);
				pendingTurns = Math.max(0, pendingTurns - 1);
				// Drain any newly queued follow-ups before deciding whether to close stdin.
				if (!steerInterrupted && !params.abortSignal?.aborted) {
					const canContinue = !parsed.malformedReason && parsed.result?.status === "SUCCESS";
					if (canContinue) {
						flushDrainedFollowUps();
					}
					if (canContinue && pendingTurns > 0) {
						const nextFollowUp = writtenFollowUps[resultsReceived - 1];
						streamedText = "";
						if (nextFollowUp !== undefined) {
							// Defer recording the intermediate turn boundary until the child actually
							// emits a frame for the successor turn; if stdin was closed and the child
							// exits now, exit recovery records the transition exactly once on resume.
							pendingIntermediateTransition = {
								prevResult: parsed.result,
								prevUsage: currentUsage(),
								followUp: nextFollowUp,
							};
						}
					} else {
						closeStdinSafely();
					}
				}
			}
			try {
				params.onEvent(parsed);
			} catch (err) {
				logger.warn("agy stream event callback threw", {
					error: formatErrorMessage(err),
				});
			}
		};

		const readStdout = cp
			.readStdout(chunk => {
				pending += decoder.decode(chunk, { stream: true });
				let nl = pending.indexOf("\n");
				while (nl !== -1) {
					const line = pending.slice(0, nl);
					pending = pending.slice(nl + 1);
					handleLine(line);
					nl = pending.indexOf("\n");
				}
			})
			.then(() => {
				const tail = pending + decoder.decode();
				pending = "";
				if (tail.trim()) handleLine(tail);
			})
			.catch(err => {
				logger.warn("agy stdout reader failed", {
					error: formatErrorMessage(err),
				});
			});

		let exitCode = 1;
		try {
			exitCode = await cp.exited;
		} catch (err) {
			if (err instanceof ptree.TimeoutError) {
				timedOut = true;
				aborted = true;
				abortReason = `Subagent exceeded task.maxRuntimeMs (${params.maxRuntimeMs}ms)`;
			} else if (steerInterrupted && !params.abortSignal?.aborted) {
				// Intentional mid-turn steer interrupt: outer loop immediately resumes via `--conversation`.
				exitCode = 0;
			} else if (err instanceof ptree.AbortError || params.abortSignal?.aborted) {
				aborted = true;
				abortReason = abortReason ?? abortReasonOf(params.abortSignal, "Cancelled");
			} else {
				exitCode = 1;
			}
		}

		await Promise.all([readStdout, cp.stderrDone.catch(() => {}), cp.drainDone.catch(() => {})]);
		if (!aborted && cp.exitReason?.aborted) {
			if (cp.exitReason instanceof ptree.TimeoutError) {
				timedOut = true;
				aborted = true;
				abortReason = `Subagent exceeded task.maxRuntimeMs (${params.maxRuntimeMs}ms)`;
			} else if (steerInterrupted && !params.abortSignal?.aborted) {
				exitCode = 0;
			} else {
				aborted = true;
				abortReason = abortReason ?? abortReasonOf(params.abortSignal, "Cancelled");
			}
		}

		stderrText = cp.peekStderr().trim();

		const unconfirmedFollowUps = collectUnconfirmedFollowUps();
		const lastResultAlreadyPublished = lastResultPublished && !activeTurnHasResult;
		if (steerInterrupted && !aborted && !timedOut) {
			return {
				exitCode: 0,
				conversationId,
				model,
				response: streamedText,
				responseIsAuthoritative: false,
				usage: currentUsage(true),
				durationMs: Math.max(0, Date.now() - stepStartTime),
				stderr: stderrText,
				aborted: false,
				timedOut: false,
				interruptedForSteer: true,
				...(unconfirmedFollowUps.length > 0 ? { unconfirmedFollowUps } : {}),
				...(lastResultAlreadyPublished ? { lastResultAlreadyPublished: true } : {}),
			};
		}

		const hasMalformedResult = resultMalformedReason !== undefined;
		const hasTerminalSuccess = !hasMalformedResult && resultEvent !== undefined && resultEvent.status === "SUCCESS";
		const statusIsError = !hasMalformedResult && resultEvent !== undefined && resultEvent.status !== "SUCCESS";
		const missingResultOnZeroExit = !aborted && !hasMalformedResult && exitCode === 0 && !hasTerminalSuccess;
		const responseIsAuthoritative = activeTurnHasResult && typeof resultEvent?.response === "string";
		const finalResponse = responseIsAuthoritative ? (resultEvent?.response as string) : streamedText;
		const durationMs =
			resultEvent?.duration_seconds !== undefined
				? Math.max(0, Math.round(resultEvent.duration_seconds * 1000))
				: Math.max(0, Date.now() - stepStartTime);
		const errorMessage =
			(typeof resultEvent?.error === "string" && resultEvent.error.length > 0 ? resultEvent.error : undefined) ||
			(aborted ? abortReason : undefined) ||
			(hasMalformedResult ? resultMalformedReason : undefined) ||
			(statusIsError ? `agy exited with status ${resultEvent?.status}` : undefined) ||
			(missingResultOnZeroExit ? "agy exited without emitting a terminal result frame" : undefined) ||
			(exitCode !== 0 ? stderrText || `agy exited with code ${exitCode}` : undefined);

		const effectiveExitCode =
			aborted || hasMalformedResult || statusIsError || missingResultOnZeroExit ? 1 : exitCode;
		return {
			exitCode: effectiveExitCode,
			conversationId,
			model,
			response: finalResponse,
			responseIsAuthoritative,
			structuredOutput: activeTurnHasResult ? resultEvent?.structured_output : undefined,
			usage: currentUsage(true),
			durationMs,
			error: errorMessage,
			stderr: stderrText || errorMessage || "",
			aborted,
			timedOut,
			abortReason,
			...(unconfirmedFollowUps.length > 0 ? { unconfirmedFollowUps } : {}),
			...(lastResultAlreadyPublished ? { lastResultAlreadyPublished: true } : {}),
		};
	} catch (err) {
		const message = formatErrorMessage(err);
		return {
			exitCode: 1,
			conversationId,
			model,
			response: streamedText,
			responseIsAuthoritative: false,
			usage: currentUsage(),
			durationMs: Math.max(0, Date.now() - stepStartTime),
			error: aborted ? (abortReason ?? message) : message,
			stderr: stderrText || message,
			aborted,
			timedOut,
			abortReason,
		};
	} finally {
		params.onProcessDetached?.();
		closeStdinSafely();
		if (cp && cp.exitCode === null && cp.exitReason === undefined) {
			cp.kill(new ptree.AbortError("agy step cleanup", cp.peekStderr()), -1);
		}
	}
}

function finalizeStructuredOutput(params: {
	outputSchema: unknown;
	outputSchemaMode: ExecutorOptions["outputSchemaMode"];
	outputSchemaSource: ExecutorOptions["outputSchemaSource"];
	exitCode: number;
	rawOutput: string;
	parsedFromAgy: unknown;
	error?: string;
}): { structuredOutput?: StructuredSubagentOutput; rawOutput: string; exitCode: number; error?: string } {
	const source = params.outputSchemaSource ?? (params.outputSchema !== undefined ? "caller" : "none");
	if (source === "none" || params.outputSchema === undefined) {
		return { rawOutput: params.rawOutput, exitCode: params.exitCode, error: params.error };
	}
	const mode = params.outputSchemaMode ?? "permissive";
	if (params.exitCode !== 0) {
		return {
			rawOutput: params.rawOutput,
			exitCode: params.exitCode,
			error: params.error,
			structuredOutput: {
				source,
				mode,
				status: "unavailable",
				...(params.error ? { error: params.error } : {}),
			},
		};
	}

	let candidate: unknown = params.parsedFromAgy;
	let parsedOk = candidate !== undefined;
	if (!parsedOk) {
		try {
			candidate = JSON.parse(params.rawOutput);
			parsedOk = true;
		} catch {}
	}

	const { validator, normalized, error: schemaErr } = buildOutputValidator(params.outputSchema);
	if (schemaErr || normalized === undefined) {
		return {
			rawOutput: params.rawOutput,
			exitCode: params.exitCode,
			error: params.error,
			structuredOutput: {
				source,
				mode,
				status: "unavailable",
				...(schemaErr ? { error: schemaErr } : {}),
			},
		};
	}

	// Valid unconstrained schema (`true` / `{}`): accept any parsed JSON payload as valid.
	if (!validator) {
		if (parsedOk) {
			const normalizedJson = JSON.stringify(candidate, null, 2);
			return {
				rawOutput: normalizedJson ?? params.rawOutput,
				exitCode: 0,
				error: params.error,
				structuredOutput: {
					source,
					mode,
					status: "valid",
					data: candidate,
				},
			};
		}
		const invalidError = "Output schema validation failed: agy response was not valid JSON";
		return {
			rawOutput: params.rawOutput,
			exitCode: mode === "strict" ? 1 : params.exitCode,
			error: mode === "strict" ? invalidError : params.error,
			structuredOutput: {
				source,
				mode,
				status: "invalid",
				error: invalidError,
			},
		};
	}

	const normalizedCandidate = parsedOk ? validator.normalize(candidate) : candidate;
	const verdict = parsedOk ? validator.validate(normalizedCandidate) : undefined;
	if (parsedOk && verdict?.success) {
		const normalizedJson = JSON.stringify(normalizedCandidate, null, 2);
		return {
			rawOutput: normalizedJson ?? params.rawOutput,
			exitCode: 0,
			error: params.error,
			structuredOutput: {
				source,
				mode,
				status: "valid",
				data: normalizedCandidate,
			},
		};
	}

	const failureSummary =
		parsedOk && verdict && !verdict.success
			? summarizeValidationFailure(verdict, normalizedCandidate, validator.requiredFields).message
			: "agy response was not valid JSON matching the output schema";
	const invalidError = `Output schema validation failed: ${failureSummary}`;
	return {
		rawOutput: params.rawOutput,
		exitCode: mode === "strict" ? 1 : params.exitCode,
		error: mode === "strict" ? invalidError : params.error,
		structuredOutput: {
			source,
			mode,
			status: "invalid",
			...(parsedOk ? { data: normalizedCandidate } : {}),
			error: invalidError,
		},
	};
}

function sanitizeTranscriptToolArgs(
	parameters: Record<string, unknown> | undefined,
	previewArgs: string,
): Record<string, unknown> {
	if (!parameters) {
		return previewArgs ? { preview: previewArgs } : {};
	}
	try {
		const serialized = JSON.stringify(parameters);
		if (serialized !== undefined && Buffer.byteLength(serialized, "utf8") <= TRANSCRIPT_MAX_TOOL_ARGS_BYTES) {
			return parameters;
		}
	} catch {
		// Fall back to bounded preview if parameters contain non-serializable values.
	}
	return previewArgs ? { preview: previewArgs, truncated: true } : { truncated: true };
}

function truncateTranscriptText(text: string): string {
	if (!text) return "";
	return truncateTail(text, {
		maxBytes: TRANSCRIPT_MAX_TOOL_OUTPUT_BYTES,
		maxLines: TRANSCRIPT_MAX_TOOL_OUTPUT_LINES,
	}).content;
}

interface AgyTranscriptUsage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

function buildTranscriptUsage(input: number, output: number, cacheRead: number): AgyTranscriptUsage {
	const safeInput = Math.max(0, input);
	const safeOutput = Math.max(0, output);
	const safeCacheRead = Math.max(0, cacheRead);
	return {
		input: safeInput,
		output: safeOutput,
		cacheRead: safeCacheRead,
		cacheWrite: 0,
		totalTokens: safeInput + safeOutput + safeCacheRead,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	};
}

/**
 * Streams a standard OMP child session `.jsonl` transcript (`<artifactsDir>/<id>.jsonl`)
 * so Herdr / Herdrup Chat view and OMP Traces can:
 * - open live and finished `agy` subagent transcripts (`has_transcript: true`),
 * - display the resolved `agy-cli/<model>` via `model_change`,
 * - show live tool intents via `tool_execution_start`,
 * - inspect per-turn `usage`, `api`, `provider`, `model`, and `duration` on assistant turns, and
 * - settle completed/failed/aborted runs via `session_exit` even when `wait` consumes `async-result`.
 *
 * The file is opened exclusively (`wx`) and written through its open `FileHandle` so an existing
 * transcript is never overwritten and unlinking/replacing the path mid-run can never recreate a
 * headerless file. All filesystem errors are isolated and disable further writes so transcript I/O
 * can never fail or reject a subagent run.
 */
class AgyTranscriptWriter {
	readonly filePath: string;
	readonly #agentId: string;
	#handle: fs.FileHandle | null;
	#queue: Promise<void> = Promise.resolve();
	#disabled = false;
	#closed = false;
	#entrySeq = 0;
	#lastEntryId: string | null = null;
	#lastModel: string;
	#recordedConversations = new Set<string>();

	private constructor(params: {
		filePath: string;
		handle: fs.FileHandle;
		agentId: string;
		initialModel: string;
		initialSeq: number;
		lastEntryId: string | null;
		recordedConversations?: Iterable<string>;
	}) {
		this.filePath = params.filePath;
		this.#handle = params.handle;
		this.#agentId = params.agentId;
		this.#lastModel = params.initialModel;
		this.#entrySeq = params.initialSeq;
		this.#lastEntryId = params.lastEntryId;
		if (params.recordedConversations) {
			for (const cid of params.recordedConversations) {
				this.#recordedConversations.add(cid);
			}
		}
	}

	static async openForAppend(params: {
		filePath: string;
		agentId: string;
		initialModel: string;
	}): Promise<AgyTranscriptWriter | undefined> {
		let initialSeq = 0;
		let lastEntryId: string | null = null;
		let lastModel = params.initialModel;
		const recordedConversations = new Set<string>();
		try {
			const existing = await fs.readFile(params.filePath, "utf8");
			for (const rawLine of existing.split("\n")) {
				const trimmed = rawLine.trim();
				if (!trimmed) continue;
				try {
					const parsed: unknown = JSON.parse(trimmed);
					if (!isRecord(parsed)) continue;
					if (typeof parsed.id === "string" && parsed.type !== "session") {
						lastEntryId = parsed.id;
						const seqMatch = /^agy-(\d+)$/.exec(parsed.id);
						if (seqMatch) {
							const n = Number.parseInt(seqMatch[1]!, 10);
							if (Number.isFinite(n) && n > initialSeq) initialSeq = n;
						}
					}
					if (parsed.type === "model_change" && typeof parsed.model === "string" && parsed.model) {
						lastModel = parsed.model;
					}
					if (
						parsed.type === "custom" &&
						parsed.customType === "agy_session" &&
						isRecord(parsed.data) &&
						typeof parsed.data.conversationId === "string" &&
						parsed.data.conversationId.trim()
					) {
						recordedConversations.add(parsed.data.conversationId.trim());
					}
				} catch {
					// Ignore malformed historical lines while scanning the tail.
				}
			}
		} catch (err) {
			logger.warn("agy transcript append scan failed", {
				id: params.agentId,
				path: params.filePath,
				error: formatErrorMessage(err),
			});
			return undefined;
		}
		let handle: fs.FileHandle | null = null;
		try {
			handle = await fs.open(params.filePath, "a", 0o600);
			return new AgyTranscriptWriter({
				filePath: params.filePath,
				handle,
				agentId: params.agentId,
				initialModel: lastModel,
				initialSeq,
				lastEntryId,
				recordedConversations,
			});
		} catch (err) {
			if (handle) {
				await handle.close().catch(() => {});
			}
			logger.warn("agy transcript reopen failed", {
				id: params.agentId,
				path: params.filePath,
				error: formatErrorMessage(err),
			});
			return undefined;
		}
	}

	static async create(params: {
		artifactsDir?: string;
		id: string;
		cwd: string;
		parentSession?: string | null;
		initialModel: string;
	}): Promise<AgyTranscriptWriter | undefined> {
		if (!params.artifactsDir) return undefined;
		if (
			!SAFE_TRANSCRIPT_ID_RE.test(params.id) ||
			params.id === "." ||
			params.id === ".." ||
			params.id.includes("..")
		) {
			return undefined;
		}
		const filePath = path.join(params.artifactsDir, `${params.id}.jsonl`);
		const nowIso = new Date().toISOString();
		const header = {
			type: "session",
			version: CURRENT_SESSION_VERSION,
			id: crypto.randomUUID(),
			timestamp: nowIso,
			cwd: params.cwd,
			...(params.parentSession ? { parentSession: params.parentSession } : {}),
		};
		const modelEntryId = "agy-1";
		const modelEntry = {
			type: "model_change",
			id: modelEntryId,
			parentId: null,
			timestamp: nowIso,
			model: params.initialModel,
		};
		const initialPayload = `${JSON.stringify(header)}\n${JSON.stringify(modelEntry)}\n`;

		let handle: fs.FileHandle | null = null;
		try {
			handle = await fs.open(filePath, "wx", 0o600);
			await handle.write(initialPayload, undefined, "utf8");
			return new AgyTranscriptWriter({
				filePath,
				handle,
				agentId: params.id,
				initialModel: params.initialModel,
				initialSeq: 1,
				lastEntryId: modelEntryId,
			});
		} catch (err) {
			if (handle) {
				await handle.close().catch(() => {});
			}
			logger.warn("agy transcript init failed", {
				id: params.id,
				path: filePath,
				error: formatErrorMessage(err),
			});
			return undefined;
		}
	}

	get activePath(): string | undefined {
		return this.#disabled ? undefined : this.filePath;
	}

	#buildEntry(type: string, timestamp: string, extra: Record<string, unknown>): Record<string, unknown> {
		const id = `agy-${++this.#entrySeq}`;
		const parentId = this.#lastEntryId;
		this.#lastEntryId = id;
		// Keep `type` and `id` as the first two keys so Herdr's 256-byte prefix intactness check hits directly.
		return { type, id, parentId, timestamp, ...extra };
	}

	#enqueue(type: string, extra: Record<string, unknown>, timestamp = new Date().toISOString()): void {
		if (this.#disabled || this.#closed || !this.#handle) return;
		const entry = this.#buildEntry(type, timestamp, extra);
		let line: string;
		try {
			line = `${JSON.stringify(entry)}\n`;
		} catch (err) {
			logger.warn("agy transcript serialize failed", {
				id: this.#agentId,
				type,
				error: formatErrorMessage(err),
			});
			return;
		}
		this.#queue = this.#queue.then(async () => {
			if (this.#disabled || !this.#handle) return;
			try {
				await this.#handle.write(line, undefined, "utf8");
			} catch (err) {
				this.#disabled = true;
				const failedHandle = this.#handle;
				this.#handle = null;
				await failedHandle.close().catch(() => {});
				logger.warn("agy transcript append failed", {
					id: this.#agentId,
					path: this.filePath,
					error: formatErrorMessage(err),
				});
			}
		});
	}

	recordModelChange(modelDisplay: string): void {
		if (!modelDisplay || modelDisplay === this.#lastModel) return;
		this.#lastModel = modelDisplay;
		this.#enqueue("model_change", { model: modelDisplay });
	}

	recordAgySession(conversationId: string): void {
		const trimmed = conversationId.trim();
		if (!trimmed || this.#recordedConversations.has(trimmed)) return;
		this.#recordedConversations.add(trimmed);
		this.#enqueue("custom", {
			customType: "agy_session",
			data: { conversationId: trimmed },
		});
	}

	recordUserPrompt(text: string): void {
		const now = Date.now();
		this.#enqueue(
			"message",
			{
				message: {
					role: "user",
					attribution: "agent",
					content: [{ type: "text", text }],
					timestamp: now,
				},
			},
			new Date(now).toISOString(),
		);
	}

	recordToolStart(params: {
		callId: string;
		toolName: string;
		toolArgs: Record<string, unknown>;
		intent: string;
		precedingText?: string;
		usage?: AgyTranscriptUsage;
		durationMs?: number;
	}): void {
		const now = Date.now();
		const iso = new Date(now).toISOString();
		const content: Array<Record<string, unknown>> = [];
		if (params.precedingText?.trim()) {
			content.push({ type: "text", text: params.precedingText.trim() });
		}
		content.push({
			type: "toolCall",
			id: params.callId,
			name: params.toolName,
			arguments: params.toolArgs,
		});
		this.#enqueue(
			"message",
			{
				message: {
					role: "assistant",
					content,
					api: "agy-cli",
					provider: "agy-cli",
					model: bareAgyModelName(this.#lastModel),
					stopReason: "toolUse",
					...(params.usage ? { usage: params.usage } : {}),
					...(params.durationMs !== undefined ? { duration: params.durationMs } : {}),
					timestamp: now,
				},
			},
			iso,
		);
		this.#enqueue(
			"custom",
			{
				customType: "tool_execution_start",
				data: {
					toolCallId: params.callId,
					toolName: params.toolName,
					startedAt: iso,
					intent: params.intent,
				},
			},
			iso,
		);
	}

	recordToolResult(params: { callId: string; toolName: string; output: string; isError?: boolean }): void {
		const now = Date.now();
		const bounded = truncateTranscriptText(params.output);
		this.#enqueue(
			"message",
			{
				message: {
					role: "toolResult",
					toolCallId: params.callId,
					toolName: params.toolName,
					isError: Boolean(params.isError),
					content: bounded ? [{ type: "text", text: bounded }] : [],
					timestamp: now,
				},
			},
			new Date(now).toISOString(),
		);
	}

	recordAssistantTurn(params: {
		text: string;
		stopReason: "stop" | "error" | "aborted";
		errorMessage?: string;
		usage?: AgyTranscriptUsage;
		durationMs?: number;
	}): void {
		const now = Date.now();
		const content: Array<Record<string, unknown>> = params.text ? [{ type: "text", text: params.text }] : [];
		this.#enqueue(
			"message",
			{
				message: {
					role: "assistant",
					content,
					api: "agy-cli",
					provider: "agy-cli",
					model: bareAgyModelName(this.#lastModel),
					stopReason: params.stopReason,
					...(params.errorMessage ? { errorMessage: params.errorMessage } : {}),
					...(params.usage ? { usage: params.usage } : {}),
					...(params.durationMs !== undefined ? { duration: params.durationMs } : {}),
					timestamp: now,
				},
			},
			new Date(now).toISOString(),
		);
	}

	async finish(params: {
		status: "completed" | "failed" | "aborted";
		text: string;
		errorMessage?: string;
		modelDisplay?: string;
		usage?: AgyTranscriptUsage;
		durationMs?: number;
	}): Promise<void> {
		if (!this.#closed) {
			if (params.modelDisplay) {
				this.recordModelChange(params.modelDisplay);
			}
			const stopReason: "stop" | "error" | "aborted" =
				params.status === "completed" ? "stop" : params.status === "aborted" ? "aborted" : "error";
			const finalText = params.text || params.errorMessage || (params.status === "aborted" ? "Cancelled" : "");
			this.recordAssistantTurn({
				text: finalText,
				stopReason,
				errorMessage: params.status === "completed" ? undefined : params.errorMessage,
				usage: params.usage,
				durationMs: params.durationMs,
			});
			const nowIso = new Date().toISOString();
			this.#enqueue(
				"custom",
				{
					customType: "session_exit",
					data: {
						reason: "dispose",
						kind: "normal",
						recordedAt: nowIso,
					},
				},
				nowIso,
			);
			this.#closed = true;
		}
		await this.close();
	}

	async close(): Promise<void> {
		this.#closed = true;
		try {
			await this.#queue;
		} finally {
			const handle = this.#handle;
			this.#handle = null;
			if (handle) {
				await handle.close().catch(() => {});
			}
		}
	}
}

const EMPTY_EXTENSION_ROOTS: EffectiveExtensionRoots = {
	explicit: [],
	mode: "explicit-only",
	configured: [],
	configuredLevel: "user",
};

function combineAbortSignals(signals: Array<AbortSignal | undefined>): {
	signal: AbortSignal;
	dispose: () => void;
} {
	const controller = new AbortController();
	const active = signals.filter((s): s is AbortSignal => s !== undefined);
	for (const sig of active) {
		if (sig.aborted) {
			controller.abort(sig.reason);
			return { signal: controller.signal, dispose: () => {} };
		}
	}
	const cleanups: Array<() => void> = [];
	for (const sig of active) {
		const onAbort = () => {
			if (!controller.signal.aborted) controller.abort(sig.reason);
		};
		sig.addEventListener("abort", onAbort, { once: true });
		cleanups.push(() => sig.removeEventListener("abort", onAbort));
	}
	return {
		signal: controller.signal,
		dispose: () => {
			for (const fn of cleanups) fn();
			cleanups.length = 0;
		},
	};
}

function isParentOrAncestor(senderId: string, parentId: string | undefined): boolean {
	if (!parentId) return senderId === MAIN_AGENT_ID;
	if (senderId === parentId) return true;
	const registry = AgentRegistry.global();
	const seen = new Set<string>([parentId]);
	for (
		let cur = registry.get(parentId)?.parentId;
		cur !== undefined && !seen.has(cur);
		cur = registry.get(cur)?.parentId
	) {
		if (cur === senderId) return true;
		seen.add(cur);
	}
	return false;
}

function formatIrcMessageForAgy(msg: IrcMessage, kind: "steer" | "followUp" | "wake"): string {
	const sender = msg.fromLabel ? `${msg.from} (${msg.fromLabel})` : msg.from;
	if (kind === "steer") {
		return `[Urgent steering message from ${sender}]:\n${msg.body}`;
	}
	if (kind === "wake") {
		return `[Follow-up message from ${sender}]:\n${msg.body}`;
	}
	const tag = msg.progress ? "Progress note" : msg.broadcast ? "Broadcast" : "Message";
	return `[${tag} from ${sender}]:\n${msg.body}`;
}

interface QueuedAgyInput {
	text: string;
	from?: string;
	relayTo?: { to: string; replyTo: string };
}

interface PendingWakeTurn {
	text: string;
	from?: string;
	relayTo?: { to: string; replyTo: string };
	resolve?: () => void;
	reject?: (err: unknown) => void;
}

interface AgyControllerConfig {
	id: string;
	binary: string;
	model?: string;
	resolvedModel: string;
	cwd: string;
	additionalDirectories?: string[];
	artifactsDir?: string;
	transcriptPath?: string;
	parentId?: string;
	maxRuntimeMs: number;
	conversationId?: string;
	initialRunning: boolean;
}

function prepareOutputSchema(outputSchema: unknown): {
	jsonSchemaStr?: string;
	jsonPromptInstruction?: string;
} {
	if (outputSchema === undefined) return {};
	const { jsonSchema, normalized, error: schemaErr } = buildOutputValidator(outputSchema);
	if (schemaErr || normalized === undefined) return {};
	if (jsonSchema && jsonSchema.type === "object") {
		// Official `agy --json-schema` requires root `"type": "object"`.
		return { jsonSchemaStr: JSON.stringify(jsonSchema) };
	}
	if (jsonSchema && Object.keys(jsonSchema).length > 0) {
		return { jsonPromptInstruction: buildJsonSchemaPromptInstruction(JSON.stringify(jsonSchema)) };
	}
	// Valid unconstrained JSON Schema (`true` or `{}`): `agy --json-schema` rejects schemas
	// without root `"type": "object"`, so explicitly request raw JSON in the final step prompt.
	return { jsonPromptInstruction: buildJsonSchemaPromptInstruction() };
}

interface AgyProgressReporter {
	appendText(delta: string): void;
	getTailText(): string;
	schedule(flush?: boolean): void;
	clearTimer(): void;
}

/** Shared rolling-tail & timer-coalescing progress reporter used by both initial and monitored follow-up turns. */
function createAgyProgressReporter(params: {
	id: string;
	progress: AgentProgress;
	startTime: number;
	onProgress?: (progress: AgentProgress) => void;
	emitFrame: (snapshot: AgentProgress) => void;
}): AgyProgressReporter {
	let recentOutputTail = "";
	let recentOutputDirty = false;
	let lastEmitMs = 0;
	let timer: ReturnType<typeof setTimeout> | null = null;

	const emitNow = () => {
		if (recentOutputDirty) {
			recentOutputDirty = false;
			const tail =
				recentOutputTail.length > RECENT_OUTPUT_TAIL_BYTES
					? recentOutputTail.slice(-RECENT_OUTPUT_TAIL_BYTES)
					: recentOutputTail;
			params.progress.recentOutput = tail
				.split("\n")
				.filter(line => line.trim())
				.slice(-RECENT_OUTPUT_LINES)
				.reverse();
		}
		params.progress.durationMs = Date.now() - params.startTime;
		const snapshot = { ...params.progress };
		if (params.onProgress) {
			invokeObserverSafely("agy onProgress callback failed", { id: params.id, status: snapshot.status }, () =>
				params.onProgress?.(snapshot),
			);
		}
		params.emitFrame(snapshot);
		lastEmitMs = Date.now();
	};

	const clearTimer = () => {
		if (timer) {
			clearTimeout(timer);
			timer = null;
		}
	};

	return {
		appendText(delta: string) {
			recentOutputTail += delta;
			if (recentOutputTail.length > RECENT_OUTPUT_TAIL_BYTES * 2) {
				recentOutputTail = recentOutputTail.slice(-RECENT_OUTPUT_TAIL_BYTES);
			}
			recentOutputDirty = true;
		},
		getTailText: () => recentOutputTail,
		schedule(flush = false) {
			if (flush) {
				clearTimer();
				emitNow();
				return;
			}
			const elapsed = Date.now() - lastEmitMs;
			if (lastEmitMs === 0 || elapsed >= PROGRESS_COALESCE_MS) {
				clearTimer();
				emitNow();
				return;
			}
			if (timer) return;
			timer = setTimeout(() => {
				timer = null;
				emitNow();
			}, PROGRESS_COALESCE_MS - elapsed);
		},
		clearTimer,
	};
}

interface TurnSequenceInput {
	steps: Array<{
		promptText: string;
		transcriptPrompt: string;
		isFinalStep: boolean;
		jsonSchema?: string;
		jsonPromptInstruction?: string;
	}>;
	abortSignal: AbortSignal;
	deadlineMs?: number;
	maxRuntimeMs: number;
	transcript?: AgyTranscriptWriter;
	progress?: AgentProgress;
	onScheduleProgress?: (flush?: boolean) => void;
	onToolResult?: ExecutorOptions["onToolResult"];
	onTextDelta?: (delta: string) => void;
	onResolvedModel?: (rawModel: string | undefined) => void;
	activityRef?: AgentRef;
	callIdNamespace: string;
}

interface TurnSequenceSummary {
	lastOutcome: StepRunOutcome;
	runtimeExceeded: boolean;
	completedRequests: number;
	completedInputTokens: number;
	completedOutputTokens: number;
	completedCacheReadTokens: number;
	finalStepUnrecordedText: string;
	finalStepResidualUsage?: AgyTranscriptUsage;
	finalStepDurationMs?: number;
}

class AgySessionController {
	readonly id: string;
	readonly binary: string;
	readonly configuredModel?: string;
	readonly cwd: string;
	readonly additionalDirectories?: string[];
	readonly artifactsDir?: string;
	readonly parentId?: string;
	readonly maxRuntimeMs: number;

	#transcriptPath?: string;
	#conversationId?: string;
	#resolvedModel: string;
	#running: boolean;
	#inActiveTurn: boolean;
	#admissionClosed = false;
	#wakeHoldCount = 0;
	#runStartedAt: number | undefined;
	#disposed = false;
	#activeProcess: ActiveAgyProcessControl | null = null;
	#activeStepPromise: Promise<unknown> | null = null;
	#runAbortController = new AbortController();
	#pendingSteers: QueuedAgyInput[] = [];
	#queuedFollowUps: QueuedAgyInput[] = [];
	#wakeQueue: PendingWakeTurn[] = [];
	#wakeLoopPromise: Promise<void> | null = null;
	#turnLock: Promise<void> = Promise.resolve();
	#pendingReplies = new Set<Promise<void>>();
	#runStateListeners = new Set<(state: "running" | "idle") => void>();
	#sessionEventListeners = new Set<AgentSessionEventListener>();
	#disposers: Array<() => void> = [];
	#messages: AgentMessage[] = [];
	#pendingToolCalls = new Set<string>();
	#userMessages = 0;
	#assistantMessages = 0;
	#toolCalls = 0;
	#toolResults = 0;
	#inputTokens = 0;
	#outputTokens = 0;
	#cacheReadTokens = 0;
	#turnStepSeq = 0;
	#sessionHandle: AgySessionHandle | undefined;

	constructor(cfg: AgyControllerConfig) {
		this.id = cfg.id;
		this.binary = cfg.binary;
		this.configuredModel = cfg.model;
		this.cwd = cfg.cwd;
		this.additionalDirectories = cfg.additionalDirectories;
		this.artifactsDir = cfg.artifactsDir;
		this.parentId = cfg.parentId;
		this.maxRuntimeMs = cfg.maxRuntimeMs;
		this.#transcriptPath = cfg.transcriptPath;
		this.#conversationId = cfg.conversationId;
		this.#resolvedModel = cfg.resolvedModel;
		this.#running = cfg.initialRunning;
		this.#inActiveTurn = cfg.initialRunning;
		this.#runStartedAt = cfg.initialRunning ? Date.now() : undefined;
	}

	get transcriptPath(): string | undefined {
		return this.#transcriptPath;
	}

	setTranscriptPath(filePath: string | undefined): void {
		this.#transcriptPath = filePath;
	}

	get conversationId(): string | undefined {
		return this.#conversationId;
	}

	setConversationId(cid: string | undefined): void {
		if (cid && cid.trim()) {
			this.#conversationId = cid.trim();
		}
	}

	get resolvedModel(): string {
		return this.#resolvedModel;
	}

	setResolvedModel(display: string): void {
		if (display) this.#resolvedModel = display;
	}

	get isRunning(): boolean {
		return this.#running;
	}

	get isInActiveTurn(): boolean {
		return this.#inActiveTurn;
	}

	get isClosedForInput(): boolean {
		return this.#disposed || this.#admissionClosed;
	}

	get runStartedAt(): number | undefined {
		return this.#runStartedAt;
	}

	get isDisposed(): boolean {
		return this.#disposed;
	}

	get abortSignal(): AbortSignal {
		return this.#runAbortController.signal;
	}

	get messages(): AgentMessage[] {
		return this.#messages;
	}

	get pendingToolCalls(): Set<string> {
		return this.#pendingToolCalls;
	}

	get hasPendingSteer(): boolean {
		return this.#pendingSteers.length > 0;
	}

	get hasPendingWakeWork(): boolean {
		return (
			this.#wakeHoldCount > 0 ||
			this.#pendingSteers.length > 0 ||
			this.#queuedFollowUps.length > 0 ||
			this.#wakeQueue.length > 0 ||
			this.#wakeLoopPromise !== null
		);
	}

	addDisposer(fn: () => void): void {
		this.#disposers.push(fn);
	}

	subscribeRunState(listener: (state: "running" | "idle") => void): () => void {
		this.#runStateListeners.add(listener);
		return () => {
			this.#runStateListeners.delete(listener);
		};
	}

	subscribeSessionEvent(listener: AgentSessionEventListener): () => void {
		this.#sessionEventListeners.add(listener);
		return () => {
			this.#sessionEventListeners.delete(listener);
		};
	}

	emitSessionEvent(event: AgentSessionEvent): void {
		for (const listener of this.#sessionEventListeners) {
			try {
				listener(event);
			} catch (err) {
				logger.warn("agy session event listener threw", {
					id: this.id,
					type: event.type,
					error: formatErrorMessage(err),
				});
			}
		}
	}

	bindSessionHandle(handle: AgySessionHandle): void {
		this.#sessionHandle = handle;
	}

	/**
	 * Single ownership-checked authority for run-state transitions: updates local state once,
	 * commits `AgentRegistry.setStatus` once (unless `skipRegistry` is set), and notifies run-state
	 * listeners once. Returns the registry CAS result (or `true` when skipped).
	 */
	commitRunState(
		state: "running" | "idle",
		options?: { expected?: AgentRef | AgentSessionHandle; skipRegistry?: boolean },
	): boolean {
		const nextRunning = state === "running";
		const stateChanged = this.#running !== nextRunning;
		if (stateChanged) {
			this.#running = nextRunning;
			this.#runStartedAt = nextRunning ? Date.now() : undefined;
			if (!nextRunning) {
				this.#pendingToolCalls.clear();
			}
		}
		let committed = true;
		if (!options?.skipRegistry) {
			const expected = options?.expected ?? this.#sessionHandle;
			if (expected) {
				committed = AgentRegistry.global().setStatus(this.id, state, expected);
			}
		}
		if (stateChanged) {
			for (const listener of this.#runStateListeners) {
				try {
					listener(state);
				} catch (err) {
					logger.warn("agy run-state listener threw", {
						id: this.id,
						state,
						error: formatErrorMessage(err),
					});
				}
			}
		}
		return committed;
	}

	/** End in-turn input routing for a retained session and commit idle once. */
	endActiveTurn(expected?: AgentRef | AgentSessionHandle): boolean {
		this.#inActiveTurn = false;
		return this.commitRunState("idle", { expected });
	}

	/** Close input admission permanently (one-shot teardown or abort). */
	closeAdmission(): void {
		this.#inActiveTurn = false;
		this.#admissionClosed = true;
		this.#pendingSteers.length = 0;
		this.#queuedFollowUps.length = 0;
		this.commitRunState("idle", { skipRegistry: true });
	}

	/** Hold wake-queue pumping while final initial-run transcript flush completes. */
	holdWakes(): () => void {
		this.#wakeHoldCount += 1;
		let released = false;
		return () => {
			if (released) return;
			released = true;
			this.#wakeHoldCount = Math.max(0, this.#wakeHoldCount - 1);
			if (this.#wakeHoldCount === 0 && this.#wakeQueue.length > 0 && !this.#disposed && !this.#admissionClosed) {
				this.#ensureWakeLoop();
			}
		};
	}

	attachProcess(control: ActiveAgyProcessControl): void {
		this.#activeProcess = control;
		if (this.#conversationId && this.#pendingSteers.length > 0) {
			control.interruptForSteer();
		}
	}

	detachProcess(): void {
		this.#activeProcess = null;
	}

	trackStepPromise(promise: Promise<unknown>): void {
		this.#activeStepPromise = promise;
		void promise.finally(() => {
			if (this.#activeStepPromise === promise) {
				this.#activeStepPromise = null;
			}
		});
	}

	recordUserMessage(text: string): void {
		this.#userMessages += 1;
		this.#messages.push({
			role: "user",
			content: [{ type: "text", text }],
			timestamp: Date.now(),
		});
	}

	recordAssistantMessage(text: string): void {
		if (!text) return;
		this.#assistantMessages += 1;
		this.#messages.push({
			role: "assistant",
			content: [{ type: "text", text }],
			api: "agy-cli" as unknown as "anthropic-messages",
			provider: "agy-cli",
			model: bareAgyModelName(this.#resolvedModel),
			usage: buildTranscriptUsage(0, 0, 0),
			stopReason: "stop",
			timestamp: Date.now(),
		});
	}

	noteToolStart(callId: string): void {
		this.#toolCalls += 1;
		this.#pendingToolCalls.add(callId);
	}

	noteToolDone(callId: string): void {
		this.#toolResults += 1;
		this.#pendingToolCalls.delete(callId);
	}

	addUsage(input: number, output: number, cacheRead: number): void {
		this.#inputTokens += Math.max(0, input);
		this.#outputTokens += Math.max(0, output);
		this.#cacheReadTokens += Math.max(0, cacheRead);
	}

	getUsageStatistics(): UsageStatistics {
		return {
			input: this.#inputTokens,
			output: this.#outputTokens,
			cacheRead: this.#cacheReadTokens,
			cacheWrite: 0,
			totalTokens: this.#inputTokens + this.#outputTokens + this.#cacheReadTokens,
			orchestrationInput: 0,
			orchestrationOutput: 0,
			orchestrationCacheRead: 0,
			premiumRequests: 0,
			cost: 0,
			subagentCost: 0,
		};
	}

	getSessionStats(): SessionStats {
		const totalTokens = this.#inputTokens + this.#outputTokens + this.#cacheReadTokens;
		return {
			sessionFile: this.#transcriptPath,
			sessionId: this.#conversationId ?? this.id,
			userMessages: this.#userMessages,
			assistantMessages: this.#assistantMessages,
			toolCalls: this.#toolCalls,
			toolResults: this.#toolResults,
			totalMessages: this.#messages.length,
			tokens: {
				input: this.#inputTokens,
				output: this.#outputTokens,
				reasoning: 0,
				cacheRead: this.#cacheReadTokens,
				cacheWrite: 0,
				total: totalTokens,
			},
			premiumRequests: 0,
			cost: 0,
		};
	}

	hasQueuedIrcFrom(from: string): boolean {
		return (
			this.#pendingSteers.some(item => item.from === from) ||
			this.#queuedFollowUps.some(item => item.from === from) ||
			this.#wakeQueue.some(item => item.from === from)
		);
	}

	enqueueFollowUp(text: string, from?: string, relayTo?: { to: string; replyTo: string }): void {
		if (this.#activeProcess && this.#pendingSteers.length === 0) {
			if (this.#activeProcess.writeFollowUp(text)) {
				return;
			}
		}
		this.#queuedFollowUps.push({ text, from, relayTo });
	}

	enqueueSteer(text: string, from?: string, relayTo?: { to: string; replyTo: string }): void {
		this.#pendingSteers.push({ text, from, relayTo });
		if (this.#activeProcess && this.#conversationId) {
			this.#activeProcess.interruptForSteer();
		}
	}

	drainQueuedFollowUps(): string[] {
		if (this.#queuedFollowUps.length === 0) return [];
		const items = this.#queuedFollowUps.splice(0);
		return items.map(item => item.text);
	}

	drainAllPendingMessages(unconfirmedFollowUps: readonly UnconfirmedFollowUp[] = []): UnconfirmedFollowUp[] {
		const steers = this.#pendingSteers.splice(0).map(item => ({
			text: item.text,
			alreadyPublishedInTranscript: false,
		}));
		const followUps = this.#queuedFollowUps.splice(0).map(item => ({
			text: item.text,
			alreadyPublishedInTranscript: false,
		}));
		return [...unconfirmedFollowUps, ...steers, ...followUps];
	}

	/** Promote any messages that arrived during initial-run teardown into wake turns so none are stranded at idle. */
	promoteLatePendingToWakeTurns(): void {
		const late = [...this.#pendingSteers.splice(0), ...this.#queuedFollowUps.splice(0)];
		for (const item of late) {
			this.scheduleWakeTurn({
				text: item.text,
				from: item.from,
				...(item.relayTo ? { relayTo: item.relayTo } : {}),
			});
		}
	}

	#ensureWakeLoop(): void {
		if (this.#wakeLoopPromise || this.#wakeHoldCount > 0 || this.#disposed || this.#admissionClosed) return;
		const loopPromise = this.#pumpWakeQueue();
		this.#wakeLoopPromise = loopPromise;
		this.#pendingReplies.add(loopPromise);
		void loopPromise.finally(() => {
			if (this.#wakeLoopPromise === loopPromise) {
				this.#wakeLoopPromise = null;
			}
			this.#pendingReplies.delete(loopPromise);
			if (this.#wakeQueue.length > 0 && !this.#disposed && !this.#admissionClosed && this.#wakeHoldCount === 0) {
				this.#ensureWakeLoop();
			}
		});
	}

	scheduleWakeTurn(item: PendingWakeTurn): void {
		this.#wakeQueue.push(item);
		this.#ensureWakeLoop();
	}

	async runForegroundPrompt(text: string): Promise<boolean> {
		if (this.#disposed || this.#admissionClosed) return false;
		const { promise, resolve, reject } = Promise.withResolvers<void>();
		this.scheduleWakeTurn({ text, resolve, reject });
		await promise;
		return true;
	}

	async #withTurnLock<T>(signal: AbortSignal | undefined, fn: () => Promise<T>): Promise<T> {
		if (signal?.aborted) {
			throw new Error(abortReasonOf(signal, "Cancelled"));
		}
		const prev = this.#turnLock;
		let release!: () => void;
		const gate = new Promise<void>(resolve => {
			release = resolve;
		});
		this.#turnLock = prev.then(() => gate);
		try {
			if (signal) {
				await new Promise<void>((resolve, reject) => {
					const onAbort = () => reject(new Error(abortReasonOf(signal, "Cancelled")));
					signal.addEventListener("abort", onAbort, { once: true });
					void prev.then(() => {
						signal.removeEventListener("abort", onAbort);
						if (signal.aborted) {
							reject(new Error(abortReasonOf(signal, "Cancelled")));
						} else {
							resolve();
						}
					});
				});
			} else {
				await prev;
			}
			return await fn();
		} finally {
			release();
		}
	}

	/**
	 * Single unified turn execution loop shared by initial runs, idle/revived IRC wake turns,
	 * and monitored `runSubagentFollowUpTurn` calls. Owns process attach/detach, single-source
	 * conversation ID publication, tool/usage streaming, unconfirmed follow-up recovery, and
	 * immediate mid-turn steer interruption/resumption.
	 */
	async executeTurnSequence(input: TurnSequenceInput): Promise<TurnSequenceSummary> {
		let runtimeExceeded = false;
		let completedRequests = 0;
		let completedInputTokens = 0;
		let completedOutputTokens = 0;
		let completedCacheReadTokens = 0;
		let finalStepUnrecordedText = "";
		let finalStepResidualUsage: AgyTranscriptUsage | undefined;
		let finalStepDurationMs: number | undefined;
		let lastOutcome: StepRunOutcome = {
			exitCode: 1,
			response: "",
			responseIsAuthoritative: false,
			usage: { requests: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0 },
			durationMs: 0,
			stderr: "",
			aborted: false,
			timedOut: false,
		};
		const registry = AgentRegistry.global();

		for (let i = 0; i < input.steps.length; i++) {
			const step = input.steps[i]!;
			let currentTranscriptPrompt = step.transcriptPrompt;
			let recordNextUserPrompt = true;
			let currentStepPrompt = step.promptText;
			let steerAttempt = 0;
			let stepFailed = false;

			while (true) {
				if (input.deadlineMs !== undefined && Date.now() >= input.deadlineMs) {
					runtimeExceeded = true;
					stepFailed = true;
					break;
				}
				if (input.abortSignal.aborted || this.#disposed) {
					stepFailed = true;
					break;
				}

				if (recordNextUserPrompt && currentTranscriptPrompt.length > 0) {
					input.transcript?.recordUserPrompt(currentTranscriptPrompt);
					this.recordUserMessage(currentTranscriptPrompt);
				}
				let liveStepRequests = 0;
				let liveStepTokens = 0;
				let recordedStepInput = 0;
				let recordedStepOutput = 0;
				let recordedStepCacheRead = 0;
				let pendingTurnUsage: { input: number; output: number; cacheRead: number; durationMs?: number } | undefined;
				let toolCallSeq = 0;
				let pendingAssistantText = "";
				const stepNumber = ++this.#turnStepSeq;
				const stepCallPrefix =
					steerAttempt === 0 ? `${input.callIdNamespace}-${i}` : `${input.callIdNamespace}-${i}-s${steerAttempt}`;
				const activeToolsByStep = new Map<
					number,
					{ callId: string; tool: string; args: string; argsKey?: string; startMs: number }
				>();
				const unindexedToolQueue: Array<{
					callId: string;
					tool: string;
					args: string;
					argsKey?: string;
					startMs: number;
				}> = [];

				const consumePendingTurnUsage = (): { usage?: AgyTranscriptUsage; durationMs?: number } => {
					if (!pendingTurnUsage) return {};
					const current = pendingTurnUsage;
					pendingTurnUsage = undefined;
					recordedStepInput += current.input;
					recordedStepOutput += current.output;
					recordedStepCacheRead += current.cacheRead;
					return {
						usage: buildTranscriptUsage(current.input, current.output, current.cacheRead),
						durationMs: current.durationMs,
					};
				};

				const stepPromise = runSingleAgyStep({
					binary: this.binary,
					cwd: this.cwd,
					model: this.configuredModel,
					conversationId: this.#conversationId,
					additionalDirectories: this.additionalDirectories,
					jsonSchema: step.isFinalStep ? step.jsonSchema : undefined,
					promptText: currentStepPrompt,
					id: this.id,
					stepIndex: stepNumber,
					abortSignal: input.abortSignal,
					deadlineMs: input.deadlineMs,
					maxRuntimeMs: input.maxRuntimeMs,
					onProcessAttached: control => this.attachProcess(control),
					onProcessDetached: () => this.detachProcess(),
					onConversationId: cid => {
						this.setConversationId(cid);
						input.transcript?.recordAgySession(cid);
					},
					onIntermediateTurnComplete: ({ prevResult, prevUsage, nextPrompt }) => {
						const prevReply =
							typeof prevResult?.response === "string" ? prevResult.response : pendingAssistantText;
						const flushed = consumePendingTurnUsage();
						let turnUsage = flushed.usage;
						let turnDurationMs = flushed.durationMs;
						const deltaIn = Math.max(0, prevUsage.inputTokens - recordedStepInput);
						const deltaOut = Math.max(0, prevUsage.outputTokens - recordedStepOutput);
						const deltaCache = Math.max(0, prevUsage.cacheReadTokens - recordedStepCacheRead);
						if (deltaIn > 0 || deltaOut > 0 || deltaCache > 0) {
							recordedStepInput += deltaIn;
							recordedStepOutput += deltaOut;
							recordedStepCacheRead += deltaCache;
							const prevIn = turnUsage?.input ?? 0;
							const prevOut = turnUsage?.output ?? 0;
							const prevCache = turnUsage?.cacheRead ?? 0;
							turnUsage = buildTranscriptUsage(prevIn + deltaIn, prevOut + deltaOut, prevCache + deltaCache);
						}
						if (turnDurationMs === undefined && prevResult?.duration_seconds !== undefined) {
							turnDurationMs = Math.max(0, Math.round(prevResult.duration_seconds * 1000));
						}
						if (prevReply || turnUsage) {
							input.transcript?.recordAssistantTurn({
								text: prevReply,
								stopReason: "stop",
								usage: turnUsage,
								durationMs: turnDurationMs,
							});
							if (prevReply) this.recordAssistantMessage(prevReply);
						}
						pendingAssistantText = "";
						if (input.progress) input.progress.contextTokens = undefined;
						input.transcript?.recordUserPrompt(nextPrompt);
						this.recordUserMessage(nextPrompt);
					},
					drainQueuedFollowUps: () => this.drainQueuedFollowUps(),
					hasPendingSteer: () => this.hasPendingSteer,
					onEvent: ev => {
						if (ev.event === "init") {
							if (ev.init?.model) {
								if (input.onResolvedModel) {
									input.onResolvedModel(ev.init.model);
								} else {
									const display = formatResolvedAgyModel(ev.init.model);
									this.setResolvedModel(display);
									input.transcript?.recordModelChange(display);
								}
								input.onScheduleProgress?.();
							}
							return;
						}
						if (ev.event === "step_update" && ev.step_update) {
							const su = ev.step_update;
							if (su.step_type === "tool") {
								const frameToolName = su.tool_name || su.tool_info?.name;
								const stepKey = su.step_index;
								if (su.state === "ACTIVE") {
									const toolName = frameToolName || "tool";
									const summarized = summarizeToolArgs(su.tool_info?.parameters, this.cwd);
									const existing = stepKey !== undefined ? activeToolsByStep.get(stepKey) : undefined;
									if (existing) {
										if (frameToolName) existing.tool = frameToolName;
										if (summarized.args) {
											existing.args = summarized.args;
											existing.argsKey = summarized.argsKey;
										}
										if (input.progress) {
											input.progress.currentTool = existing.tool;
											input.progress.currentToolArgs = existing.args;
											input.progress.currentToolArgsKey = existing.argsKey;
											input.progress.currentToolStartMs = existing.startMs;
											input.progress.streaming = undefined;
										}
										if (input.activityRef) {
											registry.setActivity(
												this.id,
												existing.args ? `${existing.tool}: ${existing.args}` : existing.tool,
												input.activityRef,
											);
										}
										input.onScheduleProgress?.();
										return;
									}
									const startedAt = Date.now();
									const callId = `${stepCallPrefix}-${++toolCallSeq}`;
									const activeEntry = {
										callId,
										tool: toolName,
										args: summarized.args,
										argsKey: summarized.argsKey,
										startMs: startedAt,
									};
									if (stepKey !== undefined) {
										activeToolsByStep.set(stepKey, activeEntry);
									} else {
										unindexedToolQueue.push(activeEntry);
									}
									const intent = summarized.args ? `${toolName}: ${summarized.args}` : toolName;
									const attached = consumePendingTurnUsage();
									this.noteToolStart(callId);
									if (input.activityRef) {
										registry.setActivity(this.id, intent, input.activityRef);
									}
									input.transcript?.recordToolStart({
										callId,
										toolName,
										toolArgs: sanitizeTranscriptToolArgs(su.tool_info?.parameters, summarized.args),
										intent,
										precedingText: pendingAssistantText,
										usage: attached.usage,
										durationMs: attached.durationMs,
									});
									pendingAssistantText = "";
									if (input.progress) {
										input.progress.currentTool = toolName;
										input.progress.currentToolArgs = summarized.args;
										input.progress.currentToolArgsKey = summarized.argsKey;
										input.progress.currentToolStartMs = startedAt;
										input.progress.streaming = undefined;
									}
									input.onScheduleProgress?.();
								} else if (su.state === "DONE") {
									let prior:
										| { callId: string; tool: string; args: string; argsKey?: string; startMs: number }
										| undefined;
									if (stepKey !== undefined) {
										prior = activeToolsByStep.get(stepKey);
										activeToolsByStep.delete(stepKey);
									} else if (frameToolName) {
										const matchIdx = unindexedToolQueue.findIndex(entry => entry.tool === frameToolName);
										if (matchIdx !== -1) {
											prior = unindexedToolQueue.splice(matchIdx, 1)[0];
										}
									} else {
										prior = unindexedToolQueue.shift();
									}
									const resolvedName = frameToolName || prior?.tool || "tool";
									const summarized = summarizeToolArgs(su.tool_info?.parameters, this.cwd);
									const resolvedArgs = summarized.args || prior?.args || "";
									const resolvedArgsKey = summarized.argsKey ?? prior?.argsKey;
									let callId = prior?.callId;
									if (!callId) {
										callId = `${stepCallPrefix}-${++toolCallSeq}`;
										const intent = resolvedArgs ? `${resolvedName}: ${resolvedArgs}` : resolvedName;
										const attached = consumePendingTurnUsage();
										this.noteToolStart(callId);
										input.transcript?.recordToolStart({
											callId,
											toolName: resolvedName,
											toolArgs: sanitizeTranscriptToolArgs(su.tool_info?.parameters, resolvedArgs),
											intent,
											precedingText: pendingAssistantText,
											usage: attached.usage,
											durationMs: attached.durationMs,
										});
										pendingAssistantText = "";
									}
									this.noteToolDone(callId);
									input.transcript?.recordToolResult({
										callId,
										toolName: resolvedName,
										output: su.tool_info?.output ?? "",
									});
									if (input.progress) {
										input.progress.toolCount += 1;
										input.progress.recentTools.push({
											tool: resolvedName,
											args: resolvedArgs,
											argsKey: resolvedArgsKey,
											endMs: Date.now(),
										});
										if (input.progress.recentTools.length > RECENT_TOOLS_LIMIT) {
											input.progress.recentTools.shift();
										}
										const remaining = activeToolsByStep.values().next().value ?? unindexedToolQueue[0];
										if (remaining) {
											input.progress.currentTool = remaining.tool;
											input.progress.currentToolArgs = remaining.args;
											input.progress.currentToolArgsKey = remaining.argsKey;
											input.progress.currentToolStartMs = remaining.startMs;
										} else {
											input.progress.currentTool = undefined;
											input.progress.currentToolArgs = undefined;
											input.progress.currentToolArgsKey = undefined;
											input.progress.currentToolStartMs = undefined;
										}
									}
									if (input.onToolResult) {
										invokeObserverSafely(
											"agy onToolResult callback failed",
											{ id: this.id, tool: resolvedName },
											() =>
												input.onToolResult?.({
													tool: resolvedName,
													text: su.tool_info?.output ?? "",
													isError: false,
												}),
										);
									}
									input.onScheduleProgress?.();
								}
							} else if (su.step_type === "agent_response") {
								if (pendingTurnUsage && (su.state === "ACTIVE" || su.text_delta || su.state === "DONE")) {
									const flushed = consumePendingTurnUsage();
									input.transcript?.recordAssistantTurn({
										text: pendingAssistantText.trim(),
										stopReason: "stop",
										usage: flushed.usage,
										durationMs: flushed.durationMs,
									});
									pendingAssistantText = "";
								}
								if (su.text_delta) {
									input.onTextDelta?.(su.text_delta);
									pendingAssistantText += su.text_delta;
								}
								if (su.state === "ACTIVE") {
									if (input.progress) input.progress.streaming = { kind: "text" };
									input.onScheduleProgress?.();
								} else if (su.state === "DONE") {
									if (input.progress) input.progress.streaming = undefined;
									liveStepRequests += 1;
									if (su.usage) {
										const inp = su.usage.input_tokens ?? 0;
										const out = su.usage.output_tokens ?? 0;
										const cacheRead = su.usage.cache_read_tokens ?? 0;
										liveStepTokens += inp + out;
										const ctx = inp + cacheRead;
										if (ctx > 0 && input.progress) input.progress.contextTokens = ctx;
										pendingTurnUsage = {
											input: inp,
											output: out,
											cacheRead,
											durationMs:
												su.duration_seconds !== undefined
													? Math.max(0, Math.round(su.duration_seconds * 1000))
													: undefined,
										};
									}
									if (input.progress) {
										input.progress.requests = completedRequests + liveStepRequests;
										input.progress.tokens = completedInputTokens + completedOutputTokens + liveStepTokens;
									}
									input.onScheduleProgress?.();
								}
							}
						}
					},
				});
				this.trackStepPromise(stepPromise);
				lastOutcome = await stepPromise;

				completedRequests += lastOutcome.usage.requests;
				completedInputTokens += lastOutcome.usage.inputTokens;
				completedOutputTokens += lastOutcome.usage.outputTokens;
				completedCacheReadTokens += lastOutcome.usage.cacheReadTokens;
				this.addUsage(
					lastOutcome.usage.inputTokens,
					lastOutcome.usage.outputTokens,
					lastOutcome.usage.cacheReadTokens,
				);
				if (input.progress) {
					input.progress.requests = completedRequests;
					input.progress.tokens = completedInputTokens + completedOutputTokens;
					if (lastOutcome.usage.lastContextTokens !== undefined) {
						input.progress.contextTokens = lastOutcome.usage.lastContextTokens;
					}
				}
				if (lastOutcome.timedOut) {
					runtimeExceeded = true;
				}
				if (lastOutcome.model) {
					if (input.onResolvedModel) {
						input.onResolvedModel(lastOutcome.model);
					} else {
						const display = formatResolvedAgyModel(lastOutcome.model);
						this.setResolvedModel(display);
						input.transcript?.recordModelChange(display);
					}
				}

				const remInput = Math.max(0, lastOutcome.usage.inputTokens - recordedStepInput);
				const remOutput = Math.max(0, lastOutcome.usage.outputTokens - recordedStepOutput);
				const remCacheRead = Math.max(0, lastOutcome.usage.cacheReadTokens - recordedStepCacheRead);
				const hasResidualUsage =
					remInput > 0 ||
					remOutput > 0 ||
					remCacheRead > 0 ||
					(recordedStepInput === 0 &&
						recordedStepOutput === 0 &&
						recordedStepCacheRead === 0 &&
						lastOutcome.usage.requests > 0);
				const stepResidualUsage = hasResidualUsage
					? buildTranscriptUsage(remInput, remOutput, remCacheRead)
					: undefined;
				const stepDurationMs = pendingTurnUsage?.durationMs ?? lastOutcome.durationMs;

				if (lastOutcome.aborted || runtimeExceeded || input.abortSignal.aborted || this.#disposed) {
					finalStepUnrecordedText = lastOutcome.responseIsAuthoritative
						? lastOutcome.response
						: pendingAssistantText;
					finalStepResidualUsage = stepResidualUsage;
					finalStepDurationMs = stepDurationMs;
					stepFailed = true;
					break;
				}

				// Recover unconfirmed stdin follow-ups (e.g. child closed fd 0) and mid-turn steers/queued input.
				const pendingNext =
					lastOutcome.exitCode === 0 || lastOutcome.interruptedForSteer
						? this.drainAllPendingMessages(lastOutcome.unconfirmedFollowUps)
						: [];
				if (lastOutcome.interruptedForSteer || pendingNext.length > 0) {
					if (!this.#conversationId) {
						const missingCidErr =
							"agy exited without a conversation ID before delivering accepted follow-up/steering message(s)";
						lastOutcome = {
							...lastOutcome,
							exitCode: 1,
							error: lastOutcome.error ?? missingCidErr,
							stderr: lastOutcome.stderr || missingCidErr,
						};
						finalStepUnrecordedText = lastOutcome.responseIsAuthoritative
							? lastOutcome.response
							: pendingAssistantText;
						finalStepResidualUsage = stepResidualUsage;
						finalStepDurationMs = stepDurationMs;
						stepFailed = true;
						break;
					}
					if (!lastOutcome.interruptedForSteer && !lastOutcome.lastResultAlreadyPublished) {
						const prevReply = lastOutcome.responseIsAuthoritative ? lastOutcome.response : pendingAssistantText;
						if (prevReply || stepResidualUsage) {
							input.transcript?.recordAssistantTurn({
								text: prevReply,
								stopReason: "stop",
								usage: stepResidualUsage,
								durationMs: stepDurationMs,
							});
							this.recordAssistantMessage(prevReply);
						}
					}
					if (pendingNext.length > 0) {
						const unrecordedPrompts = pendingNext
							.filter(item => !item.alreadyPublishedInTranscript)
							.map(item => item.text);
						const combinedPrompt = pendingNext.map(item => item.text).join("\n\n");
						currentTranscriptPrompt = unrecordedPrompts.join("\n\n");
						recordNextUserPrompt = unrecordedPrompts.length > 0;
						currentStepPrompt =
							step.isFinalStep && step.jsonPromptInstruction
								? `${combinedPrompt}\n\n${step.jsonPromptInstruction}`
								: combinedPrompt;
						steerAttempt += 1;
						continue;
					}
				}

				if (lastOutcome.exitCode !== 0) {
					finalStepUnrecordedText = lastOutcome.responseIsAuthoritative
						? lastOutcome.response
						: pendingAssistantText;
					finalStepResidualUsage = stepResidualUsage;
					finalStepDurationMs = stepDurationMs;
					stepFailed = true;
					break;
				}

				if (!step.isFinalStep) {
					const stepReplyText = lastOutcome.responseIsAuthoritative ? lastOutcome.response : pendingAssistantText;
					if (stepReplyText || stepResidualUsage) {
						input.transcript?.recordAssistantTurn({
							text: stepReplyText,
							stopReason: "stop",
							usage: stepResidualUsage,
							durationMs: stepDurationMs,
						});
						this.recordAssistantMessage(stepReplyText);
					}
					pendingAssistantText = "";
					finalStepUnrecordedText = "";
					finalStepResidualUsage = undefined;
					finalStepDurationMs = undefined;
				} else {
					finalStepUnrecordedText = lastOutcome.responseIsAuthoritative
						? lastOutcome.response
						: pendingAssistantText;
					finalStepResidualUsage = stepResidualUsage;
					finalStepDurationMs = stepDurationMs;
				}
				break;
			}
			if (stepFailed) break;
		}

		return {
			lastOutcome,
			runtimeExceeded,
			completedRequests,
			completedInputTokens,
			completedOutputTokens,
			completedCacheReadTokens,
			finalStepUnrecordedText,
			finalStepResidualUsage,
			finalStepDurationMs,
		};
	}

	runWithTurnLock<T>(fn: () => Promise<T>): Promise<T> {
		return this.#withTurnLock(undefined, fn);
	}

	async runMonitoredFollowUpTurn(options: FollowUpTurnOptions, startTime: number): Promise<SingleResult> {
		if (options.restore || options.steps !== undefined) {
			throw new Error("External agy CLI subagents do not support restore/stepped-resume follow-up turns.");
		}
		if (options.topicBoard !== undefined) {
			throw new Error("External agy CLI subagents do not support topicBoard follow-up turns.");
		}
		if (options.workPoolYieldItems && options.workPoolYieldItems.length > 0) {
			throw new Error("External agy CLI subagents do not support workPoolYieldItems follow-up turns.");
		}
		if (options.completionProbe === true) {
			throw new Error("External agy CLI subagents do not support completionProbe follow-up turns.");
		}
		if (options.softRequestBudget !== undefined && options.softRequestBudget > 0) {
			throw new Error("External agy CLI subagents do not support softRequestBudget follow-up turns.");
		}
		const index = options.index ?? 0;
		if (this.#disposed || this.#admissionClosed) {
			return {
				index,
				id: options.id,
				agent: options.agent.name,
				agentSource: options.agent.source,
				task: options.message,
				description: options.description,
				exitCode: 1,
				output: "",
				stderr: "agy subagent session is disposed",
				truncated: false,
				durationMs: Math.max(0, Date.now() - startTime),
				tokens: 0,
				requests: 0,
				resolvedModel: this.#resolvedModel,
				resolvedModelIdentity: this.#resolvedModel,
				error: "agy subagent session is disposed",
			};
		}

		let postLockEmit: (() => void) | undefined;
		try {
			const result = await this.#withTurnLock(options.signal, async () => {
				if (this.#disposed || this.#admissionClosed) {
					return {
						index,
						id: options.id,
						agent: options.agent.name,
						agentSource: options.agent.source,
						task: options.message,
						description: options.description,
						exitCode: 1,
						output: "",
						stderr: "agy subagent session is disposed",
						truncated: false,
						durationMs: Math.max(0, Date.now() - startTime),
						tokens: 0,
						requests: 0,
						resolvedModel: this.#resolvedModel,
						resolvedModelIdentity: this.#resolvedModel,
						error: "agy subagent session is disposed",
					};
				}
				if (this.#runAbortController.signal.aborted) {
					this.#runAbortController = new AbortController();
				}
				const combinedAbort = combineAbortSignals([options.signal, this.#runAbortController.signal]);
				const maxRuntimeMs = Math.max(0, Math.trunc(Number(options.maxRuntimeMs ?? this.maxRuntimeMs) || 0));
				const deadlineMs = maxRuntimeMs > 0 ? startTime + maxRuntimeMs : undefined;
				const { jsonSchemaStr, jsonPromptInstruction } = prepareOutputSchema(options.outputSchema);

				const progress: AgentProgress = {
					index,
					id: options.id,
					agent: options.agent.name,
					agentSource: options.agent.source,
					status: "running",
					task: options.message,
					description: options.description,
					recentTools: [],
					recentOutput: [],
					toolCount: 0,
					requests: 0,
					tokens: 0,
					cost: 0,
					durationMs: 0,
					modelRole: options.modelRole,
					resolvedModel: this.#resolvedModel,
					resolvedModelIdentity: this.#resolvedModel,
				};
				const progressReporter = createAgyProgressReporter({
					id: this.id,
					progress,
					startTime,
					onProgress: options.onProgress,
					emitFrame: snapshot => {
						emitSubagentFrameSafely(
							options.eventBus,
							options.subagentEventBus,
							TASK_SUBAGENT_PROGRESS_CHANNEL,
							{
								index,
								agent: options.agent.name,
								agentSource: options.agent.source,
								task: options.message,
								parentToolCallId: options.parentToolCallId,
								detached: true,
								...(this.#transcriptPath ? { sessionFile: this.#transcriptPath } : {}),
								progress: snapshot,
							},
							{ id: this.id, status: snapshot.status },
						);
					},
				});

				let transcript: AgyTranscriptWriter | undefined;
				this.#inActiveTurn = true;
				this.commitRunState("running");
				this.emitSessionEvent({ type: "agent_start" });
				emitSubagentFrameSafely(
					options.eventBus,
					options.subagentEventBus,
					TASK_SUBAGENT_LIFECYCLE_CHANNEL,
					{
						id: options.id,
						agent: options.agent.name,
						parentToolCallId: options.parentToolCallId,
						detached: true,
						agentSource: options.agent.source,
						description: options.description,
						...(this.#transcriptPath ? { sessionFile: this.#transcriptPath } : {}),
						status: "started" as const,
						index,
					},
					{ id: this.id, status: "started" },
				);
				progressReporter.schedule(true);

				const releaseWakeHold = this.holdWakes();
				try {
					if (this.#transcriptPath) {
						transcript = await AgyTranscriptWriter.openForAppend({
							filePath: this.#transcriptPath,
							agentId: this.id,
							initialModel: this.#resolvedModel,
						});
					}
					const preparedSteps = [
						{
							transcriptPrompt: options.message,
							promptText: jsonPromptInstruction
								? `${options.message}\n\n${jsonPromptInstruction}`
								: options.message,
							isFinalStep: true,
							jsonSchema: jsonSchemaStr,
							jsonPromptInstruction,
						},
					];
					const summary = await this.executeTurnSequence({
						steps: preparedSteps,
						abortSignal: combinedAbort.signal,
						deadlineMs,
						maxRuntimeMs,
						transcript,
						progress,
						onScheduleProgress: flush => progressReporter.schedule(flush),
						onTextDelta: delta => progressReporter.appendText(delta),
						onResolvedModel: rawModel => {
							if (!rawModel) return;
							const display = formatResolvedAgyModel(rawModel);
							progress.resolvedModel = display;
							progress.resolvedModelIdentity = display;
							this.setResolvedModel(display);
							transcript?.recordModelChange(display);
						},
						activityRef: AgentRegistry.global().get(this.id),
						callIdNamespace: `agy-fu-${++this.#turnStepSeq}`,
					});

					// Atomically commit monitored turn acceptance vs cancellation against the registry
					// BEFORE async artifact/transcript I/O and terminal progress callbacks so RPC
					// `cancel_subagent` can never return true while this turn reports completion.
					const registry = AgentRegistry.global();
					const refAtCommit = registry.get(this.id);
					let wasAborted =
						summary.runtimeExceeded ||
						summary.lastOutcome.aborted ||
						Boolean(combinedAbort.signal.aborted) ||
						refAtCommit?.status === "aborted";
					const ownsLiveRef = Boolean(
						refAtCommit && refAtCommit.session === this.#sessionHandle && refAtCommit.status !== "aborted",
					);
					this.#inActiveTurn = false;
					// Always commit idle for an owned live ref — including when the caller aborted only this
					// turn via AbortSignal or maxRuntimeMs — so the retained session returns to idle and
					// rearms its idle TTL. Never overwrite an aborted tombstone or replacement ref.
					const committedIdle = this.commitRunState("idle", {
						expected: ownsLiveRef ? refAtCommit : this.#sessionHandle,
						skipRegistry: !ownsLiveRef,
					});
					if (!wasAborted && ownsLiveRef && !committedIdle) {
						wasAborted = true;
					}
					this.promoteLatePendingToWakeTurns();

					const abortReason = wasAborted
						? summary.runtimeExceeded
							? `Subagent exceeded task.maxRuntimeMs (${maxRuntimeMs}ms)`
							: (summary.lastOutcome.abortReason ?? abortReasonOf(combinedAbort.signal, "Cancelled"))
						: undefined;
					let exitCode = wasAborted ? 1 : summary.lastOutcome.exitCode;
					let error = wasAborted ? (abortReason ?? summary.lastOutcome.error) : summary.lastOutcome.error;
					let rawOutput = typeof summary.lastOutcome.response === "string" ? summary.lastOutcome.response : "";
					let structuredOutput: StructuredSubagentOutput | undefined;
					let truncatedOutput = rawOutput;
					let truncated = false;
					let outputPath: string | undefined;
					let outputMeta: { lineCount: number; charCount: number } | undefined;

					try {
						const structured = finalizeStructuredOutput({
							outputSchema: options.outputSchema,
							outputSchemaMode: options.outputSchemaMode,
							outputSchemaSource: options.outputSchemaSource,
							exitCode,
							rawOutput,
							parsedFromAgy: summary.lastOutcome.structuredOutput,
							error,
						});
						rawOutput = typeof structured.rawOutput === "string" ? structured.rawOutput : "";
						exitCode = structured.exitCode;
						error = structured.error;
						structuredOutput = structured.structuredOutput;

						const truncatedResult = truncateTail(rawOutput, {
							maxBytes: MAX_OUTPUT_BYTES,
							maxLines: MAX_OUTPUT_LINES,
						});
						truncatedOutput = truncatedResult.content;
						truncated = Boolean(truncatedResult.truncated);

						// Honor FollowUpTurnOptions.artifactsDir: rewrite artifacts only on valid structured
						// output or a successful non-empty message turn; preserve prior artifacts otherwise.
						const shouldPersistArtifacts =
							options.artifactsDir !== undefined &&
							!wasAborted &&
							(structuredOutput?.status === "valid" || (exitCode === 0 && rawOutput.trim().length > 0));
						if (shouldPersistArtifacts && options.artifactsDir) {
							const persisted = await persistSubagentOutputArtifacts({
								artifactsDir: options.artifactsDir,
								id: options.id,
								rawOutput,
								structuredOutput,
							});
							outputPath = persisted.outputPath;
							outputMeta = persisted.outputMeta;
						}
					} catch (err) {
						const message = formatErrorMessage(err);
						exitCode = 1;
						error = error ?? message;
					}

					progress.currentTool = undefined;
					progress.currentToolArgs = undefined;
					progress.currentToolArgsKey = undefined;
					progress.currentToolStartMs = undefined;
					progress.streaming = undefined;
					progress.status = wasAborted ? "aborted" : exitCode === 0 ? "completed" : "failed";

					const finalTranscriptText =
						summary.lastOutcome.responseIsAuthoritative || structuredOutput?.status === "valid"
							? truncatedOutput
							: truncateTail(summary.finalStepUnrecordedText, {
									maxBytes: MAX_OUTPUT_BYTES,
									maxLines: MAX_OUTPUT_LINES,
								}).content;
					if (truncatedOutput) {
						this.recordAssistantMessage(truncatedOutput);
					}
					await transcript?.finish({
						status: progress.status,
						text: finalTranscriptText,
						errorMessage: error,
						modelDisplay: progress.resolvedModel,
						usage: summary.finalStepResidualUsage,
						durationMs: summary.finalStepDurationMs,
					});

					if (outputPath) {
						invokeObserverSafely("agy follow-up setHistory failed", { id: this.id }, () => {
							registry.setHistory(this.id, { outputPath });
						});
					}

					const finalStatus = progress.status;
					postLockEmit = () => {
						progressReporter.schedule(true);
						emitSubagentFrameSafely(
							options.eventBus,
							options.subagentEventBus,
							TASK_SUBAGENT_LIFECYCLE_CHANNEL,
							{
								id: options.id,
								agent: options.agent.name,
								parentToolCallId: options.parentToolCallId,
								detached: true,
								agentSource: options.agent.source,
								description: options.description,
								...(this.#transcriptPath ? { sessionFile: this.#transcriptPath } : {}),
								status: finalStatus,
								index,
							},
							{ id: this.id, status: finalStatus },
						);
						this.emitSessionEvent({ type: "agent_end", messages: [...this.#messages] });
						releaseWakeHold();
					};

					return {
						index,
						id: options.id,
						agent: options.agent.name,
						agentSource: options.agent.source,
						task: options.message,
						description: options.description,
						exitCode,
						output: truncatedOutput,
						stderr: summary.lastOutcome.stderr || error || "",
						truncated,
						...(structuredOutput ? { structuredOutput } : {}),
						durationMs: Math.max(0, Date.now() - startTime),
						tokens: summary.completedInputTokens + summary.completedOutputTokens,
						requests: summary.completedRequests,
						contextTokens: progress.contextTokens,
						modelRole: options.modelRole,
						resolvedModel: progress.resolvedModel,
						resolvedModelIdentity: progress.resolvedModelIdentity,
						error,
						...(wasAborted ? { aborted: true, abortReason } : {}),
						usage: {
							input: summary.completedInputTokens,
							output: summary.completedOutputTokens,
							cacheRead: summary.completedCacheReadTokens,
							cacheWrite: 0,
							totalTokens:
								summary.completedInputTokens + summary.completedOutputTokens + summary.completedCacheReadTokens,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
						outputPath,
						outputMeta,
					};
				} finally {
					progressReporter.clearTimer();
					this.#inActiveTurn = false;
					combinedAbort.dispose();
					await transcript?.close().catch(() => {});
					if (!postLockEmit) {
						this.emitSessionEvent({ type: "agent_end", messages: [...this.#messages] });
						if (this.#running && (this.#wakeQueue.length === 0 || this.#disposed)) {
							const refNow = AgentRegistry.global().get(this.id);
							const ownsNow = Boolean(
								refNow && refNow.session === this.#sessionHandle && refNow.status !== "aborted",
							);
							this.commitRunState("idle", {
								expected: ownsNow ? refNow : this.#sessionHandle,
								skipRegistry: this.#disposed || !ownsNow,
							});
						}
						releaseWakeHold();
					}
				}
			});
			postLockEmit?.();
			return result;
		} catch (err) {
			if (options.signal?.aborted) {
				const reason = abortReasonOf(options.signal, "Cancelled");
				return {
					index,
					id: options.id,
					agent: options.agent.name,
					agentSource: options.agent.source,
					task: options.message,
					description: options.description,
					exitCode: 1,
					output: "",
					stderr: reason,
					truncated: false,
					durationMs: Math.max(0, Date.now() - startTime),
					tokens: 0,
					requests: 0,
					modelRole: options.modelRole,
					resolvedModel: this.#resolvedModel,
					resolvedModelIdentity: this.#resolvedModel,
					error: reason,
					aborted: true,
					abortReason: reason,
				};
			}
			throw err;
		}
	}

	async waitForIrcReplies(): Promise<void> {
		while (this.#pendingReplies.size > 0) {
			await Promise.all(this.#pendingReplies);
		}
	}

	async abort(reason?: string): Promise<void> {
		this.#pendingSteers.length = 0;
		this.#queuedFollowUps.length = 0;
		const droppedWakes = this.#wakeQueue.splice(0);
		const abortErr = new Error(reason || "Cancelled");
		for (const wake of droppedWakes) {
			wake.reject?.(abortErr);
		}
		if (!this.#runAbortController.signal.aborted) {
			this.#runAbortController.abort(reason || "Cancelled");
		}
		const activeStep = this.#activeStepPromise;
		if (activeStep) {
			await activeStep.catch(() => {});
		}
		const wakeLoop = this.#wakeLoopPromise;
		if (wakeLoop) {
			await wakeLoop.catch(() => {});
		}
	}

	async dispose(): Promise<void> {
		if (this.#disposed) return;
		this.#disposed = true;
		this.#inActiveTurn = false;
		this.#admissionClosed = true;
		for (const fn of this.#disposers.splice(0)) {
			try {
				fn();
			} catch {
				// Best-effort disposer cleanup.
			}
		}
		await this.abort("Disposed");
		this.#runStateListeners.clear();
		this.#sessionEventListeners.clear();
	}

	async #pumpWakeQueue(): Promise<void> {
		while (this.#wakeQueue.length > 0 && !this.#disposed && !this.#admissionClosed) {
			const item = this.#wakeQueue.shift()!;
			await this.#withTurnLock(undefined, async () => {
				if (this.#disposed || this.#admissionClosed) {
					item.reject?.(new Error("agy subagent session is disposed"));
					return;
				}
				if (this.#runAbortController.signal.aborted) {
					this.#runAbortController = new AbortController();
				}
				this.#inActiveTurn = true;
				this.commitRunState("running");
				this.emitSessionEvent({ type: "agent_start" });
				let wakeTranscript: AgyTranscriptWriter | undefined;
				let finalReply = "";
				let wakeError: string | undefined;
				let wakeStatus: "completed" | "failed" | "aborted" = "completed";
				try {
					if (this.#transcriptPath) {
						wakeTranscript = await AgyTranscriptWriter.openForAppend({
							filePath: this.#transcriptPath,
							agentId: this.id,
							initialModel: this.#resolvedModel,
						});
					}
					const deadlineMs = this.maxRuntimeMs > 0 ? Date.now() + this.maxRuntimeMs : undefined;
					const summary = await this.executeTurnSequence({
						steps: [
							{
								promptText: item.text,
								transcriptPrompt: item.text,
								isFinalStep: true,
							},
						],
						abortSignal: this.#runAbortController.signal,
						deadlineMs,
						maxRuntimeMs: this.maxRuntimeMs,
						transcript: wakeTranscript,
						callIdNamespace: `agy-wake-${++this.#turnStepSeq}`,
					});
					const wasAborted =
						summary.runtimeExceeded || summary.lastOutcome.aborted || this.#runAbortController.signal.aborted;
					finalReply = summary.lastOutcome.response;
					if (wasAborted) {
						wakeStatus = "aborted";
						wakeError = summary.lastOutcome.abortReason ?? summary.lastOutcome.error ?? "Cancelled";
					} else if (summary.lastOutcome.exitCode !== 0) {
						wakeStatus = "failed";
						wakeError = summary.lastOutcome.error ?? `agy exited with code ${summary.lastOutcome.exitCode}`;
					} else {
						wakeStatus = "completed";
					}
					if (finalReply) {
						this.recordAssistantMessage(finalReply);
					}
					await wakeTranscript?.finish({
						status: wakeStatus,
						text: finalReply,
						errorMessage: wakeError,
						modelDisplay: this.#resolvedModel,
						usage: summary.finalStepResidualUsage,
						durationMs: summary.finalStepDurationMs,
					});

					// Stop accepting in-turn input BEFORE emitting `agent_end` or relaying so any
					// synchronous reply/message delivered inside `agent_end` enters `#wakeQueue`
					// and runs on the very next loop iteration instead of being stranded at idle.
					this.#inActiveTurn = false;
					this.promoteLatePendingToWakeTurns();

					if (item.relayTo && !this.#disposed) {
						const body =
							wakeStatus === "completed" && finalReply.trim()
								? finalReply
								: wakeError
									? `[agy subagent ${wakeStatus}: ${wakeError}]`
									: finalReply || `[agy subagent ${wakeStatus}]`;
						try {
							await IrcBus.global().send({
								from: this.id,
								to: item.relayTo.to,
								body,
								replyTo: item.relayTo.replyTo,
								wakeRelay: true,
							});
						} catch (err) {
							logger.warn("agy wake relay failed", {
								id: this.id,
								to: item.relayTo.to,
								error: formatErrorMessage(err),
							});
						}
					}
					item.resolve?.();
				} catch (err) {
					await wakeTranscript?.close().catch(() => {});
					item.reject?.(err);
					logger.warn("agy wake turn failed", { id: this.id, error: formatErrorMessage(err) });
				} finally {
					this.#inActiveTurn = false;
					this.promoteLatePendingToWakeTurns();
					this.emitSessionEvent({ type: "agent_end", messages: [...this.#messages] });
					if (this.#wakeQueue.length === 0 || this.#disposed) {
						const refNow = AgentRegistry.global().get(this.id);
						const ownsNow = Boolean(
							refNow && refNow.session === this.#sessionHandle && refNow.status !== "aborted",
						);
						this.commitRunState("idle", {
							expected: ownsNow ? refNow : this.#sessionHandle,
							skipRegistry: this.#disposed || !ownsNow,
						});
					}
				}
			});
		}
	}
}

class AgySessionHandle implements AgentSessionHandle {
	readonly instanceKey: string;
	readonly #controller: AgySessionController;
	readonly agent: AgentSessionLoopHandle;
	readonly sessionManager: AgentSessionManagerHandle;
	readonly model: Model | undefined = undefined;
	readonly thinkingLevel: ThinkingLevel | undefined = undefined;
	readonly servingModel: ServingModel | undefined = undefined;
	readonly effectiveExtensionRoots: EffectiveExtensionRoots = EMPTY_EXTENSION_ROOTS;
	readonly hindsightActive = false;

	constructor(controller: AgySessionController) {
		this.#controller = controller;
		controller.bindSessionHandle(this);
		// Distinct prefix from `${currentThreadId()}:` so `isLocalAgentSession()` never mistakes this
		// external-CLI handle for a concrete in-process `AgentSession` instance, while registering it
		// in the local bridged-handle directory so worker threads can still read main's registry refs.
		this.instanceKey = `agy:${crypto.randomUUID()}`;
		registerLocalBridgedHandle(this);
		this.agent = {
			isPaused: false,
			state: {
				get messages() {
					return controller.messages;
				},
				streamMessage: null,
				get pendingToolCalls() {
					return controller.pendingToolCalls;
				},
			},
		};
		this.sessionManager = {
			getCwd: () => controller.cwd,
			getSessionId: () => controller.conversationId ?? controller.id,
			getSessionFile: () => controller.transcriptPath,
			getArtifactsDir: () => controller.artifactsDir ?? null,
			getUsageStatistics: () => controller.getUsageStatistics(),
		};
	}

	get isStreaming(): boolean {
		return this.#controller.isRunning;
	}

	get runStartedAt(): number | undefined {
		return this.#controller.runStartedAt;
	}

	get ircWakesPending(): boolean {
		return this.#controller.hasPendingWakeWork;
	}

	get hasPendingWake(): boolean {
		return this.#controller.isRunning || this.#controller.hasPendingWakeWork;
	}

	get messages(): AgentMessage[] {
		return this.#controller.messages;
	}

	get sessionFile(): string | undefined {
		return this.#controller.transcriptPath;
	}

	subscribe(listener: AgentSessionEventListener): () => void {
		return this.#controller.subscribeSessionEvent(listener);
	}

	subscribeRunState(listener: (state: "running" | "idle") => void): () => void {
		return this.#controller.subscribeRunState(listener);
	}

	getSessionStats(): SessionStats {
		return this.#controller.getSessionStats();
	}

	getAdvisorStatusOverview(): AgentSessionAdvisorOverview {
		return { configured: false, advisors: [] };
	}

	getSessionAgents(): AgentDefinition[] {
		return [];
	}

	hasQueuedIrcFrom(from: string): boolean {
		return this.#controller.hasQueuedIrcFrom(from);
	}

	hasIrcInterrupts(): boolean {
		return this.#controller.hasPendingSteer;
	}

	emitIrcRelayObservation(record: CustomMessage): void {
		this.#controller.emitSessionEvent({ type: "irc_message", message: record });
	}

	async deliverIrcMessage(msg: IrcMessage): Promise<"injected" | "woken"> {
		if (this.#controller.isClosedForInput) {
			throw new Error("Recipient agy subagent session is closed.");
		}
		const shouldRelay = msg.wakeRelay !== true && msg.broadcast !== true && msg.progress !== true;
		const relayTo = shouldRelay ? { to: msg.from, replyTo: msg.id } : undefined;
		if (this.#controller.isInActiveTurn) {
			const directParent =
				!msg.progress && !msg.broadcast && isParentOrAncestor(msg.from, this.#controller.parentId);
			if (directParent) {
				this.#controller.enqueueSteer(formatIrcMessageForAgy(msg, "steer"), msg.from, relayTo);
			} else {
				this.#controller.enqueueFollowUp(formatIrcMessageForAgy(msg, "followUp"), msg.from, relayTo);
			}
			return "injected";
		}
		if (msg.noWake === true) {
			this.#controller.recordUserMessage(formatIrcMessageForAgy(msg, "followUp"));
			return "injected";
		}
		this.#controller.scheduleWakeTurn({
			text: formatIrcMessageForAgy(msg, "wake"),
			from: msg.from,
			...(relayTo ? { relayTo } : {}),
		});
		return "woken";
	}

	async waitForIrcReplies(): Promise<void> {
		await this.#controller.waitForIrcReplies();
	}

	async prompt(text: string, options?: PromptOptions): Promise<boolean> {
		if (this.#controller.isClosedForInput) {
			if (options?.throwOnDrop) {
				throw new Error("agy subagent session is closed");
			}
			return false;
		}
		if (this.#controller.isInActiveTurn) {
			if (options?.streamingBehavior === "followUp" || options?.streamingBehavior === "aside") {
				this.#controller.enqueueFollowUp(text);
			} else {
				this.#controller.enqueueSteer(text);
			}
			options?.onPromptAdmitted?.();
			return true;
		}
		options?.onPromptAdmitted?.();
		return this.#controller.runForegroundPrompt(text);
	}

	async steer(text: string, _images?: ImageContent[], _options?: SteerOptions): Promise<void> {
		if (this.#controller.isClosedForInput) {
			throw new Error("agy subagent session is closed");
		}
		if (this.#controller.isInActiveTurn) {
			this.#controller.enqueueSteer(text);
			return;
		}
		await this.prompt(text);
	}

	async abort(options?: {
		goalReason?: "interrupted" | "internal";
		reason?: string;
		preserveCompaction?: boolean;
		deliverQueuedMessages?: boolean;
		detachTools?: boolean;
	}): Promise<void> {
		await this.#controller.abort(options?.reason);
	}

	async applyAgentModelChange(_change: AgentModelChange): Promise<AgentModelSelectionState> {
		throw new Error("External agy CLI subagents do not support live model changes.");
	}

	async setWorkPoolYieldItems(_items: readonly WorkPoolYieldItem[]): Promise<void> {
		// Workpool items stay on native OMP; no-op here.
	}

	async dispose(_options?: AgentSessionDisposeOptions): Promise<void> {
		await this.#controller.dispose();
	}

	runFollowUpTurn(options: FollowUpTurnOptions, startTime: number): Promise<SingleResult> {
		return this.#controller.runMonitoredFollowUpTurn(options, startTime);
	}
}

export function isAgySessionHandle(session: AgentSessionHandle): session is AgySessionHandle {
	return session instanceof AgySessionHandle;
}

function createAgyReviver(params: {
	id: string;
	binary: string;
	model?: string;
	resolvedModel: string;
	cwd: string;
	additionalDirectories?: string[];
	artifactsDir?: string;
	transcriptPath?: string;
	parentId?: string;
	maxRuntimeMs: number;
	conversationId: string;
}): AgentReviver {
	return async (_expected: AgentRef): Promise<AgentSessionHandle> => {
		const controller = new AgySessionController({
			...params,
			initialRunning: false,
		});
		return new AgySessionHandle(controller);
	};
}

/**
 * Execute a subagent via the official `agy --input-format stream-json --output-format stream-json` CLI while
 * preserving OMP's full `SingleResult`, `AgentProgress`, HUD lifecycle, mid-job messaging/steering, revival,
 * and artifact contracts.
 */
export async function runAgySubagent(options: ExecutorOptions, config: AgySubagentConfig): Promise<SingleResult> {
	const {
		cwd,
		agent,
		task,
		assignment,
		index,
		id,
		worktree,
		modelOverride,
		modelRole,
		outputSchema,
		signal,
		onProgress,
	} = options;
	const startTime = Date.now();
	const runCwd = worktree ?? cwd;
	const settings = options.settings ?? Settings.isolated();
	const maxRuntimeMs = Math.max(0, Math.trunc(Number(options.maxRuntimeMs ?? cfgTaskMaxRuntimeMs.get(settings)) || 0));

	if (signal?.aborted) {
		return {
			index,
			id,
			agent: agent.name,
			agentSource: agent.source,
			task,
			assignment,
			description: options.description,
			exitCode: 1,
			output: "",
			stderr: "Cancelled before start",
			truncated: false,
			durationMs: 0,
			tokens: 0,
			requests: 0,
			modelOverride,
			modelRole,
			resolvedModel: formatResolvedAgyModel(config.model),
			resolvedModelIdentity: formatResolvedAgyModel(config.model),
			error: "Cancelled before start",
			aborted: true,
			abortReason: "Cancelled before start",
		};
	}

	const initialResolvedModel = formatResolvedAgyModel(config.model);
	// Create the child transcript header on disk before announcing start so observers
	// (e.g. Herdr Chat view) immediately see `has_transcript: true` and the resolved model.
	const transcript = await AgyTranscriptWriter.create({
		artifactsDir: options.artifactsDir,
		id,
		cwd: runCwd,
		parentSession: options.sessionFile,
		initialModel: initialResolvedModel,
	});
	const additionalDirs = worktree !== undefined ? undefined : options.additionalDirectories;
	const controller = new AgySessionController({
		id,
		binary: config.binary,
		model: config.model,
		resolvedModel: initialResolvedModel,
		cwd: runCwd,
		additionalDirectories: additionalDirs,
		artifactsDir: options.artifactsDir,
		transcriptPath: transcript?.activePath,
		parentId: options.parentAgentId,
		maxRuntimeMs,
		initialRunning: true,
	});
	const sessionHandle = new AgySessionHandle(controller);
	const registry = AgentRegistry.global();
	let registeredRef!: AgentRef;
	let onReleaseTransferred = false;

	const progress: AgentProgress = {
		index,
		id,
		agent: agent.name,
		agentSource: agent.source,
		status: "running",
		task,
		assignment,
		description: options.description,
		recentTools: [],
		recentOutput: [],
		toolCount: 0,
		requests: 0,
		tokens: 0,
		cost: 0,
		durationMs: 0,
		modelOverride,
		modelRole,
		resolvedModel: initialResolvedModel,
		resolvedModelIdentity: initialResolvedModel,
	};

	const updateResolvedModel = (rawModel: string | undefined) => {
		if (!rawModel) return;
		const display = formatResolvedAgyModel(rawModel);
		progress.resolvedModel = display;
		progress.resolvedModelIdentity = display;
		controller.setResolvedModel(display);
		transcript?.recordModelChange(display);
	};

	const progressReporter = createAgyProgressReporter({
		id,
		progress,
		startTime,
		onProgress,
		emitFrame: snapshot => {
			const activeSessionFile = transcript?.activePath;
			emitSubagentFrameSafely(
				options.eventBus,
				options.subagentEventBus,
				TASK_SUBAGENT_PROGRESS_CHANNEL,
				{
					index,
					agent: agent.name,
					agentSource: agent.source,
					task,
					parentToolCallId: options.parentToolCallId,
					detached: options.detached,
					assignment,
					...(activeSessionFile ? { sessionFile: activeSessionFile } : {}),
					progress: snapshot,
				},
				{ id, status: snapshot.status },
			);
		},
	});

	const deadlineMs = maxRuntimeMs > 0 ? startTime + maxRuntimeMs : undefined;
	let summary: TurnSequenceSummary = {
		lastOutcome: {
			exitCode: 1,
			response: "",
			responseIsAuthoritative: false,
			usage: { requests: 0, inputTokens: 0, outputTokens: 0, cacheReadTokens: 0 },
			durationMs: 0,
			stderr: "",
			aborted: false,
			timedOut: false,
		},
		runtimeExceeded: false,
		completedRequests: 0,
		completedInputTokens: 0,
		completedOutputTokens: 0,
		completedCacheReadTokens: 0,
		finalStepUnrecordedText: "",
	};
	let wasAborted = false;
	let abortReason: string | undefined;
	let exitCode = 1;
	let error: string | undefined;
	let structuredOutput: StructuredSubagentOutput | undefined;
	let truncatedOutput = "";
	let truncated = false;
	let outputPath: string | undefined;
	let outputMeta: { lineCount: number; charCount: number } | undefined;
	let releaseWakeHold: (() => void) | undefined;

	// Reserve the controller's exclusive per-session turn lock BEFORE registering the live
	// session in AgentRegistry, and hold it across startup parked-IRC drain, initial execution,
	// atomic registry commit, artifact persistence, and `transcript.finish()` so no monitored
	// follow-up (even one started synchronously on `registered`) can ever jump ahead of the
	// initial turn or race into `<id>.jsonl` / `artifactsDir`.
	await controller.runWithTurnLock(async () => {
		registeredRef = registry.register({
			id,
			displayName: id,
			kind: "sub",
			parentId: options.parentAgentId,
			parentSessionKey: options.parentSessionKey,
			session: sessionHandle,
			sessionFile: transcript?.activePath ?? null,
			status: "running",
			activity: options.description,
		});
		const unsubRegistryAbort = registry.onChange(event => {
			if (event.type === "status_changed" && event.ref === registeredRef && event.ref.status === "aborted") {
				void controller.abort("Cancelled");
			}
		});
		controller.addDisposer(unsubRegistryAbort);
		if (registeredRef.status === "aborted") {
			void controller.abort("Cancelled");
		}
		// Drain any IRC messages parked before this subagent started running.
		const existingBus = IrcBus.existing();
		for (let parked = existingBus?.take(id); parked; parked = existingBus?.take(id)) {
			await sessionHandle.deliverIrcMessage(parked);
		}

		const combinedAbort = combineAbortSignals([signal, controller.abortSignal]);

		// Emit lifecycle start event so OMP's Subagents (Alt+A) HUD registers the worker immediately.
		const initialSessionFile = transcript?.activePath;
		emitSubagentFrameSafely(
			options.eventBus,
			options.subagentEventBus,
			TASK_SUBAGENT_LIFECYCLE_CHANNEL,
			{
				id,
				agent: agent.name,
				parentToolCallId: options.parentToolCallId,
				detached: options.detached,
				agentSource: agent.source,
				description: options.description,
				...(initialSessionFile ? { sessionFile: initialSessionFile } : {}),
				status: "started" as const,
				index,
			},
			{ id, status: "started" },
		);
		progressReporter.schedule(true);

		try {
			const { jsonSchemaStr, jsonPromptInstruction } = prepareOutputSchema(outputSchema);
			const stepPrompts = options.steps && options.steps.length > 0 ? options.steps : [task];
			const preparedSteps = stepPrompts.map((stepPrompt, i) => {
				const isFinalStep = i === stepPrompts.length - 1;
				const basePrompt = buildAgyPrompt(options, stepPrompt, i === 0);
				return {
					transcriptPrompt: stepPrompt,
					promptText:
						isFinalStep && jsonPromptInstruction ? `${basePrompt}\n\n${jsonPromptInstruction}` : basePrompt,
					isFinalStep,
					jsonSchema: isFinalStep ? jsonSchemaStr : undefined,
					jsonPromptInstruction: isFinalStep ? jsonPromptInstruction : undefined,
				};
			});
			summary = await controller.executeTurnSequence({
				steps: preparedSteps,
				abortSignal: combinedAbort.signal,
				deadlineMs,
				maxRuntimeMs,
				transcript,
				progress,
				onScheduleProgress: flush => progressReporter.schedule(flush),
				onToolResult: options.onToolResult,
				onTextDelta: delta => progressReporter.appendText(delta),
				onResolvedModel: updateResolvedModel,
				activityRef: registeredRef,
				callIdNamespace: "agy-call",
			});
		} catch (err) {
			const message = formatErrorMessage(err);
			summary = {
				...summary,
				lastOutcome: {
					...summary.lastOutcome,
					exitCode: 1,
					error: summary.lastOutcome.error ?? message,
					stderr: summary.lastOutcome.stderr || message,
				},
			};
		} finally {
			progressReporter.clearTimer();
			combinedAbort.dispose();
		}

		const { lastOutcome, runtimeExceeded, finalStepUnrecordedText, finalStepResidualUsage, finalStepDurationMs } =
			summary;

		// Atomically commit the terminal registry transition BEFORE any async artifact/transcript I/O or
		// terminal progress callbacks so RPC `cancel_subagent` can never succeed while the same run reports
		// completion, and one-shot teardown can never falsely acknowledge new mid-job messages.
		const refAtCommit = registry.get(id);
		wasAborted =
			runtimeExceeded ||
			lastOutcome.aborted ||
			Boolean(combinedAbort.signal.aborted) ||
			registeredRef.status === "aborted";
		const adoptConversationId =
			!wasAborted &&
			options.keepAlive !== false &&
			worktree === undefined &&
			controller.conversationId !== undefined &&
			refAtCommit === registeredRef
				? controller.conversationId
				: undefined;

		if (adoptConversationId !== undefined) {
			releaseWakeHold = controller.holdWakes();
			if (controller.endActiveTurn(registeredRef)) {
				AgentLifecycleManager.global().adopt(
					id,
					{
						idleTtlMs: cfgTaskAgentIdleTtlMs.get(settings),
						revive: createAgyReviver({
							id,
							binary: config.binary,
							model: config.model,
							resolvedModel: progress.resolvedModel ?? initialResolvedModel,
							cwd: runCwd,
							additionalDirectories: additionalDirs,
							artifactsDir: options.artifactsDir,
							transcriptPath: transcript?.activePath,
							parentId: options.parentAgentId,
							maxRuntimeMs,
							conversationId: adoptConversationId,
						}),
						onRelease: options.onRelease,
					},
					registeredRef,
				);
				onReleaseTransferred = true;
				controller.promoteLatePendingToWakeTurns();
			} else {
				// An explicit cancel won the transition to `aborted`.
				wasAborted = true;
				releaseWakeHold();
				releaseWakeHold = undefined;
				controller.closeAdmission();
			}
		} else {
			controller.closeAdmission();
			if (!wasAborted && refAtCommit === registeredRef && registeredRef.status !== "aborted") {
				registry.unregister(id, registeredRef);
			} else if (registeredRef.status === "aborted") {
				wasAborted = true;
			}
		}

		abortReason = wasAborted
			? runtimeExceeded
				? `Subagent exceeded task.maxRuntimeMs (${maxRuntimeMs}ms)`
				: (lastOutcome.abortReason ?? abortReasonOf(combinedAbort.signal, "Cancelled"))
			: undefined;
		exitCode = wasAborted ? 1 : lastOutcome.exitCode;
		error = wasAborted ? (abortReason ?? lastOutcome.error) : lastOutcome.error;

		let rawOutput = typeof lastOutcome.response === "string" ? lastOutcome.response : "";
		const tailText = progressReporter.getTailText().trim();
		if (wasAborted && !rawOutput.trim() && tailText.length > 0) {
			const snippet = previewLine(tailText, 200);
			rawOutput = `[cancelled after ${progress.requests} req, ${progress.tokens} tok — last activity: "${snippet}"]`;
		}

		truncatedOutput = rawOutput;
		try {
			const structured = finalizeStructuredOutput({
				outputSchema,
				outputSchemaMode: options.outputSchemaMode,
				outputSchemaSource: options.outputSchemaSource,
				exitCode,
				rawOutput,
				parsedFromAgy: lastOutcome.structuredOutput,
				error,
			});
			rawOutput = typeof structured.rawOutput === "string" ? structured.rawOutput : "";
			exitCode = structured.exitCode;
			error = structured.error;
			structuredOutput = structured.structuredOutput;

			const truncatedResult = truncateTail(rawOutput, {
				maxBytes: MAX_OUTPUT_BYTES,
				maxLines: MAX_OUTPUT_LINES,
			});
			truncatedOutput = truncatedResult.content;
			truncated = Boolean(truncatedResult.truncated);

			if (options.artifactsDir) {
				const persisted = await persistSubagentOutputArtifacts({
					artifactsDir: options.artifactsDir,
					id,
					rawOutput,
					structuredOutput,
				});
				outputPath = persisted.outputPath;
				outputMeta = persisted.outputMeta;
			}
		} catch (err) {
			const message = formatErrorMessage(err);
			exitCode = 1;
			error = error ?? message;
		}

		progress.currentTool = undefined;
		progress.currentToolArgs = undefined;
		progress.currentToolArgsKey = undefined;
		progress.currentToolStartMs = undefined;
		progress.streaming = undefined;
		progress.status = wasAborted ? "aborted" : exitCode === 0 ? "completed" : "failed";

		// Flush final assistant turn and `session_exit` to `<id>.jsonl` before releasing the turn lock
		// or queued wake turns so observers and subsequent follow-ups see a completely settled transcript.
		const finalTranscriptText =
			lastOutcome.responseIsAuthoritative || structuredOutput?.status === "valid"
				? truncatedOutput
				: truncateTail(finalStepUnrecordedText, { maxBytes: MAX_OUTPUT_BYTES, maxLines: MAX_OUTPUT_LINES }).content;
		await transcript?.finish({
			status: progress.status,
			text: finalTranscriptText,
			errorMessage: error,
			modelDisplay: progress.resolvedModel,
			usage: finalStepResidualUsage,
			durationMs: finalStepDurationMs,
		});

		const finalSessionFile = transcript?.activePath;
		controller.setTranscriptPath(finalSessionFile);
		if (truncatedOutput) {
			controller.recordAssistantMessage(truncatedOutput);
		}
	});

	const { lastOutcome, completedInputTokens, completedOutputTokens, completedCacheReadTokens } = summary;
	const finalSessionFile = transcript?.activePath;
	progressReporter.schedule(true);
	emitSubagentFrameSafely(
		options.eventBus,
		options.subagentEventBus,
		TASK_SUBAGENT_LIFECYCLE_CHANNEL,
		{
			id,
			agent: agent.name,
			parentToolCallId: options.parentToolCallId,
			detached: options.detached,
			agentSource: agent.source,
			description: progress.description,
			...(finalSessionFile ? { sessionFile: finalSessionFile } : {}),
			status: progress.status,
			index,
		},
		{ id, status: progress.status },
	);

	invokeObserverSafely("agy setHistory failed", { id }, () => {
		registry.setHistory(id, {
			agent: agent.name,
			modelRole,
			resolvedModel: progress.resolvedModel,
			...(outputPath ? { outputPath } : {}),
		});
	});

	releaseWakeHold?.();

	if (!onReleaseTransferred) {
		await sessionHandle.dispose();
		if (registry.get(id) === registeredRef && registeredRef.status !== "aborted") {
			registry.unregister(id, registeredRef);
		}
		if (options.onRelease) {
			try {
				await options.onRelease();
			} catch (err) {
				logger.warn("agy onRelease callback failed", { id, error: formatErrorMessage(err) });
			}
		}
	}

	const durationMs = Date.now() - startTime;
	return {
		index,
		id,
		agent: agent.name,
		agentSource: agent.source,
		task,
		assignment,
		description: progress.description,
		exitCode,
		output: truncatedOutput,
		stderr: lastOutcome.stderr || error || "",
		truncated,
		...(structuredOutput ? { structuredOutput } : {}),
		durationMs,
		tokens: progress.tokens,
		requests: progress.requests,
		contextTokens: progress.contextTokens,
		contextWindow: progress.contextWindow,
		modelOverride,
		modelRole,
		resolvedModel: progress.resolvedModel,
		resolvedModelIdentity: progress.resolvedModelIdentity,
		error,
		...(wasAborted ? { aborted: true, abortReason } : {}),
		usage: {
			input: completedInputTokens,
			output: completedOutputTokens,
			cacheRead: completedCacheReadTokens,
			cacheWrite: 0,
			totalTokens: completedInputTokens + completedOutputTokens + completedCacheReadTokens,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		outputPath,
		outputMeta,
	};
}
