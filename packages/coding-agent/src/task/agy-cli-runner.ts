/**
 * Runs delegated OMP subagents via the official `agy -p --output-format stream-json` CLI.
 *
 * Unlike OMP's ACP provider (which proxies OMP's own prompt/tools over `agy acp`), this runner
 * delegates the whole subagent assignment to the official `agy` CLI so `agy` runs normally with
 * its own prompt and tools while streaming live progress, tool calls, usage, artifacts, and
 * structured results straight into OMP's `Subagents` (`Alt+A`) HUD, `wait`, and `proc://`.
 */

import * as fsSync from "node:fs";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { $env, $which, isRecord, logger } from "@oh-my-pi/pi-utils";
import { AgentRegistry } from "../registry/agent-registry";
import { writeArtifact } from "../session/artifacts";
import { Settings } from "../config/settings";
import { buildOutputValidator, summarizeValidationFailure } from "../tools/output-schema-validator";
import { truncateTail } from "../tools/streaming-output";
import { emitSubagentFrame } from "../utils/event-bus";
import { previewLine, shortenToolArgumentPaths, TRUNCATE_LENGTHS } from "../utils/render-utils";
import { isReadOnlyAgent } from "./read-only-policy";
import { cfgTaskAgyBinary, cfgTaskAgySubagents, cfgTaskMaxRuntimeMs, validateAgySubagents } from "./settings";
import type { AgentProgress, SingleResult, StructuredSubagentOutput } from "./task-details";
import {
	MAX_OUTPUT_BYTES,
	MAX_OUTPUT_LINES,
	TASK_SUBAGENT_LIFECYCLE_CHANNEL,
	TASK_SUBAGENT_PROGRESS_CHANNEL,
} from "./types";
import type { ExecutorOptions } from "./executor";

const PROGRESS_COALESCE_MS = 500;
const KILL_GRACE_MS = 3_000;
const RECENT_TOOLS_LIMIT = 10;
const RECENT_OUTPUT_LINES = 8;
/** Keep argv strings safely below Linux MAX_ARG_STRLEN (128 KiB). */
const MAX_INLINE_PROMPT_BYTES = 96 * 1024;
const DISABLED_VALUES = new Set(["off", "false", "none", "disabled", "0"]);
const DEFAULT_MODEL_VALUES = new Set(["default", "auto", "true", "on", "1"]);
const MODEL_EFFORT_SUFFIX_RE = /-(low|medium|high|xhigh|max)$/;

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

export function parseAgyStreamLine(line: string): AgyStreamEvent | undefined {
	const trimmed = line.trim();
	if (!trimmed.startsWith("{")) return undefined;
	try {
		const parsed: unknown = JSON.parse(trimmed);
		if (!isRecord(parsed) || typeof parsed.event !== "string") return undefined;
		if (parsed.event === "init" || parsed.event === "step_update" || parsed.event === "result") {
			return parsed as unknown as AgyStreamEvent;
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

function resolveAgyBinaryPath(settings: Settings): string | undefined {
	const configured = cfgTaskAgyBinary.get(settings)?.trim() || $env.OMP_AGY_SUBAGENT_BIN?.trim();
	if (configured) {
		if (configured.includes("/") || configured.includes("\\")) {
			return isExecutableFile(configured) ? configured : undefined;
		}
		return $which(configured) ?? undefined;
	}
	return $which("agy") ?? undefined;
}

function applyCoarseEffort(model: string, effort: ExecutorOptions["effort"]): string {
	if (!effort || MODEL_EFFORT_SUFFIX_RE.test(model)) return model;
	if (model.startsWith("gpt-oss-") || model.startsWith("gemini-pro-orchestrator")) return model;
	const suffix = effort === "lo" ? "low" : effort === "med" ? "medium" : "high";
	return `${model}-${suffix}`;
}

/**
 * Decide whether a subagent spawn should execute via the official `agy -p` CLI.
 * Returns `undefined` to fall back cleanly to OMP's native in-process subagent executor.
 */
export function resolveAgySubagentConfig(options: ExecutorOptions): AgySubagentConfig | undefined {
	const settings = options.settings;
	if (!settings) return undefined;

	// Preserve strict OMP-only capabilities and explicit caller model pins.
	if (options.perCallModelGrants !== undefined) return undefined;
	if (options.restrictToolNames === true) return undefined;
	if (options.customTools !== undefined && options.customTools.length > 0) return undefined;
	if (options.workPoolYieldItems !== undefined && options.workPoolYieldItems.length > 0) return undefined;
	if (options.topicBoard !== undefined) return undefined;

	const mapping = validateAgySubagents(cfgTaskAgySubagents.get(settings));
	const agentName = options.agent.name;
	const rawSelector = Object.hasOwn(mapping, agentName)
		? mapping[agentName]
		: Object.hasOwn(mapping, "*")
			? mapping["*"]
			: undefined;
	if (!rawSelector) return undefined;

	const normalized = rawSelector.trim();
	const lower = normalized.toLowerCase();
	if (!normalized || DISABLED_VALUES.has(lower)) return undefined;

	const binary = resolveAgyBinaryPath(settings);
	if (!binary) {
		logger.warn("task.agySubagents configured, but `agy` binary was not found; falling back to native OMP subagent", {
			agent: agentName,
		});
		return undefined;
	}

	const model = DEFAULT_MODEL_VALUES.has(lower) ? undefined : applyCoarseEffort(normalized, options.effort);
	return { binary, model };
}

export function buildAgyPrompt(options: ExecutorOptions, stepPrompt: string, includeHeader: boolean): string {
	const sections: string[] = [];
	if (includeHeader) {
		const roleLines: string[] = [
			`You are executing a delegated "${options.agent.name}" subagent task inside a larger workflow.`,
			"Return a concise, accurate final answer directly addressing the assignment (no meta-commentary about being a subagent).",
		];
		if (isReadOnlyAgent(options.agent)) {
			roleLines.push(
				"IMPORTANT: This agent role is READ-ONLY. Do NOT create, edit, or delete repository files and do NOT run mutating commands.",
			);
		}
		sections.push(roleLines.join("\n"));

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

function inferContextWindow(model: string | undefined): number | undefined {
	if (!model) return undefined;
	if (model.startsWith("gpt-oss-")) return 131_072;
	if (model.startsWith("gemini-") || model.startsWith("claude-")) return 1_000_000;
	return undefined;
}

function formatResolvedAgyModel(model: string | undefined): string {
	return model ? `agy-cli/${model}` : "agy-cli/default";
}

function abortReasonOf(signal: AbortSignal | undefined, fallback: string): string {
	if (!signal?.aborted) return fallback;
	const reason = signal.reason;
	if (typeof reason === "string" && reason.trim()) return reason.trim();
	if (reason instanceof Error && reason.message.trim()) return reason.message.trim();
	return fallback;
}

function buildStepArgs(params: {
	binary: string;
	model?: string;
	conversationId?: string;
	additionalDirectories?: string[];
	jsonSchema?: string;
	promptArg: string;
}): string[] {
	const args = [params.binary, "--output-format", "stream-json", "--dangerously-skip-permissions"];
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
	// Attach prompt with `=` so prompts beginning with `-` are never parsed as flags.
	args.push(`--print=${params.promptArg}`);
	return args;
}

interface StepRunOutcome {
	exitCode: number;
	conversationId?: string;
	model?: string;
	response: string;
	structuredOutput?: unknown;
	usage?: AgyUsage;
	numTurns?: number;
	error?: string;
	stderr: string;
	aborted: boolean;
	abortReason?: string;
}

async function runSingleAgyStep(params: {
	binary: string;
	cwd: string;
	model?: string;
	conversationId?: string;
	additionalDirectories?: string[];
	jsonSchema?: string;
	promptText: string;
	artifactsDir?: string;
	id: string;
	stepIndex: number;
	abortSignal: AbortSignal;
	onEvent: (event: AgyStreamEvent) => void;
}): Promise<StepRunOutcome> {
	let spillPromptPath: string | undefined;
	let promptArg = params.promptText;
	if (Buffer.byteLength(params.promptText, "utf8") > MAX_INLINE_PROMPT_BYTES) {
		const spillDir = params.artifactsDir ?? os.tmpdir();
		spillPromptPath = path.join(spillDir, `${params.id}.agy-prompt-${params.stepIndex}-${Date.now()}.txt`);
		await fs.writeFile(spillPromptPath, params.promptText, { mode: 0o600 });
		promptArg = `Read the complete task instructions from ${spillPromptPath} and execute them thoroughly.`;
	}

	const cmd = buildStepArgs({
		binary: params.binary,
		model: params.model,
		conversationId: params.conversationId,
		additionalDirectories: params.additionalDirectories,
		jsonSchema: params.jsonSchema,
		promptArg,
	});

	let proc: ReturnType<typeof Bun.spawn> | undefined;
	let killTimer: ReturnType<typeof setTimeout> | undefined;
	let aborted = false;
	let abortReason: string | undefined;

	const terminateChild = () => {
		if (!proc) return;
		try {
			proc.kill("SIGTERM");
		} catch {}
		if (!killTimer) {
			killTimer = setTimeout(() => {
				try {
					proc?.kill("SIGKILL");
				} catch {}
			}, KILL_GRACE_MS);
		}
	};

	const onAbort = () => {
		aborted = true;
		abortReason = abortReasonOf(params.abortSignal, "Cancelled");
		terminateChild();
	};

	if (params.abortSignal.aborted) {
		onAbort();
		if (spillPromptPath) await fs.rm(spillPromptPath, { force: true }).catch(() => {});
		return {
			exitCode: 1,
			response: "",
			error: abortReason,
			stderr: abortReason ?? "Cancelled",
			aborted: true,
			abortReason,
		};
	}

	params.abortSignal.addEventListener("abort", onAbort, { once: true });

	let conversationId = params.conversationId;
	let model = params.model;
	let streamedText = "";
	let resultEvent: AgyStreamResult["result"] | undefined;
	let stderrText = "";

	try {
		proc = Bun.spawn(cmd, {
			cwd: params.cwd,
			stdin: "ignore",
			stdout: "pipe",
			stderr: "pipe",
			env: { ...Bun.env, AGY_CLI_DISABLE_AUTO_UPDATE: "1" },
		});

		const stdoutStream = proc.stdout;
		const stderrStream = proc.stderr;

		const readStderr = (async () => {
			if (!stderrStream || typeof stderrStream === "number") return;
			const reader = stderrStream.getReader();
			const decoder = new TextDecoder();
			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					stderrText += decoder.decode(value, { stream: true });
				}
				stderrText += decoder.decode();
			} catch {}
		})();

		const readStdout = (async () => {
			if (!stdoutStream || typeof stdoutStream === "number") return;
			const reader = stdoutStream.getReader();
			const decoder = new TextDecoder();
			let pending = "";
			const handleLine = (rawLine: string) => {
				const parsed = parseAgyStreamLine(rawLine);
				if (!parsed) return;
				if (parsed.event === "init") {
					if (parsed.conversation_id) conversationId = parsed.conversation_id;
					if (parsed.init?.model) model = parsed.init.model;
				} else if (parsed.event === "step_update") {
					if (parsed.step_update?.conversation_id) conversationId = parsed.step_update.conversation_id;
					if (parsed.step_update?.step_type === "agent_response" && parsed.step_update.text_delta) {
						streamedText += parsed.step_update.text_delta;
					}
				} else if (parsed.event === "result") {
					resultEvent = parsed.result;
					if (parsed.result?.conversation_id) conversationId = parsed.result.conversation_id;
				}
				params.onEvent(parsed);
			};
			try {
				while (true) {
					const { done, value } = await reader.read();
					if (done) break;
					pending += decoder.decode(value, { stream: true });
					let nl = pending.indexOf("\n");
					while (nl !== -1) {
						handleLine(pending.slice(0, nl));
						pending = pending.slice(nl + 1);
						nl = pending.indexOf("\n");
					}
				}
				pending += decoder.decode();
				if (pending.trim()) handleLine(pending);
			} catch {}
		})();

		const exitCode = await proc.exited;
		await Promise.allSettled([readStdout, readStderr]);

		const statusIsError = resultEvent?.status !== undefined && resultEvent.status !== "SUCCESS";
		const finalResponse = resultEvent?.response !== undefined ? resultEvent.response : streamedText;
		const errorMessage =
			resultEvent?.error ||
			(aborted ? abortReason : undefined) ||
			(statusIsError ? `agy exited with status ${resultEvent?.status}` : undefined) ||
			(exitCode !== 0 ? stderrText.trim() || `agy exited with code ${exitCode}` : undefined);

		return {
			exitCode: aborted ? 1 : statusIsError ? 1 : exitCode,
			conversationId,
			model,
			response: finalResponse,
			structuredOutput: resultEvent?.structured_output,
			usage: resultEvent?.usage,
			numTurns: resultEvent?.num_turns,
			error: errorMessage,
			stderr: stderrText.trim() || errorMessage || "",
			aborted,
			abortReason,
		};
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		return {
			exitCode: 1,
			conversationId,
			model,
			response: streamedText,
			error: aborted ? (abortReason ?? message) : message,
			stderr: stderrText.trim() || message,
			aborted,
			abortReason,
		};
	} finally {
		params.abortSignal.removeEventListener("abort", onAbort);
		if (killTimer) clearTimeout(killTimer);
		if (spillPromptPath) {
			await fs.rm(spillPromptPath, { force: true }).catch(() => {});
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

	const { validator, error: schemaErr } = buildOutputValidator(params.outputSchema);
	if (!validator || schemaErr) {
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
			...(parsedOk ? { data: candidate } : {}),
			error: invalidError,
		},
	};
}

/**
 * Execute a subagent via the official `agy -p --output-format stream-json` CLI while
 * preserving OMP's full `SingleResult`, `AgentProgress`, HUD lifecycle, and artifact contracts.
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

	if (signal?.aborted || options.launchSignal?.aborted) {
		options.onFirstTurn?.({ kind: "cancelled" });
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
		contextWindow: inferContextWindow(config.model),
		cost: 0,
		durationMs: 0,
		modelOverride,
		modelRole,
		resolvedModel: initialResolvedModel,
		resolvedModelIdentity: initialResolvedModel,
	};

	let lastProgressEmitMs = 0;
	let progressTimer: ReturnType<typeof setTimeout> | null = null;
	const recentTextChunks: string[] = [];

	const refreshRecentOutput = () => {
		const combined = recentTextChunks.join("");
		const lines = combined
			.split("\n")
			.map(l => l.trimEnd())
			.filter(l => l.length > 0);
		progress.recentOutput = lines.slice(-RECENT_OUTPUT_LINES);
	};

	const emitProgressNow = () => {
		refreshRecentOutput();
		progress.durationMs = Date.now() - startTime;
		onProgress?.({ ...progress });
		const progressPayload = {
			index,
			agent: agent.name,
			agentSource: agent.source,
			task,
			parentToolCallId: options.parentToolCallId,
			detached: options.detached,
			assignment,
			progress: { ...progress },
		};
		emitSubagentFrame(options.eventBus, options.subagentEventBus, TASK_SUBAGENT_PROGRESS_CHANNEL, progressPayload);
		lastProgressEmitMs = Date.now();
	};

	const scheduleProgress = (flush = false) => {
		if (flush) {
			if (progressTimer) {
				clearTimeout(progressTimer);
				progressTimer = null;
			}
			emitProgressNow();
			return;
		}
		const now = Date.now();
		const elapsed = now - lastProgressEmitMs;
		if (lastProgressEmitMs === 0 || elapsed >= PROGRESS_COALESCE_MS) {
			if (progressTimer) {
				clearTimeout(progressTimer);
				progressTimer = null;
			}
			emitProgressNow();
			return;
		}
		if (progressTimer) return;
		progressTimer = setTimeout(() => {
			progressTimer = null;
			emitProgressNow();
		}, PROGRESS_COALESCE_MS - elapsed);
	};

	// Emit lifecycle start event so OMP's Subagents (Alt+A) HUD registers the worker immediately.
	emitSubagentFrame(options.eventBus, options.subagentEventBus, TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
		id,
		agent: agent.name,
		parentToolCallId: options.parentToolCallId,
		detached: options.detached,
		agentSource: agent.source,
		description: options.description,
		status: "started" as const,
		index,
	});
	scheduleProgress(true);

	const runAbort = new AbortController();
	let runtimeExceeded = false;
	const forwardAbort = (src: AbortSignal | undefined, fallback: string) => () => {
		if (!runAbort.signal.aborted) {
			runAbort.abort(abortReasonOf(src, fallback));
		}
	};
	const onSignalAbort = forwardAbort(signal, "Cancelled");
	const onLaunchAbort = forwardAbort(options.launchSignal, "Parent transcript replaced");
	signal?.addEventListener("abort", onSignalAbort, { once: true });
	options.launchSignal?.addEventListener("abort", onLaunchAbort, { once: true });

	const runtimeTimer =
		maxRuntimeMs > 0
			? setTimeout(() => {
					runtimeExceeded = true;
					if (!runAbort.signal.aborted) {
						runAbort.abort(`Subagent exceeded task.maxRuntimeMs (${maxRuntimeMs}ms)`);
					}
				}, maxRuntimeMs)
			: undefined;

	let jsonSchemaStr: string | undefined;
	if (outputSchema !== undefined) {
		const { jsonSchema } = buildOutputValidator(outputSchema);
		if (jsonSchema !== undefined) {
			jsonSchemaStr = JSON.stringify(jsonSchema);
		}
	}

	const stepPrompts = options.steps && options.steps.length > 0 ? options.steps : [task];
	const additionalDirs = worktree !== undefined ? undefined : options.additionalDirectories;
	let conversationId: string | undefined;
	let lastOutcome: StepRunOutcome = {
		exitCode: 1,
		response: "",
		stderr: "",
		aborted: false,
	};
	let totalInputTokens = 0;
	let totalOutputTokens = 0;
	let totalCacheReadTokens = 0;
	let activeToolName: string | undefined;
	let activeToolArgs = "";
	let activeToolArgsKey: string | undefined;
	let firstTurnNotified = false;

	try {
		for (let i = 0; i < stepPrompts.length; i++) {
			const isFinalStep = i === stepPrompts.length - 1;
			const fullPrompt = buildAgyPrompt(options, stepPrompts[i]!, i === 0);

			lastOutcome = await runSingleAgyStep({
				binary: config.binary,
				cwd: runCwd,
				model: config.model,
				conversationId,
				additionalDirectories: additionalDirs,
				jsonSchema: isFinalStep ? jsonSchemaStr : undefined,
				promptText: fullPrompt,
				artifactsDir: options.artifactsDir,
				id,
				stepIndex: i,
				abortSignal: runAbort.signal,
				onEvent: ev => {
					if (ev.event === "init") {
						if (!firstTurnNotified) {
							firstTurnNotified = true;
							options.onFirstTurn?.({ kind: "started" });
						}
						if (ev.init?.model) {
							const display = formatResolvedAgyModel(ev.init.model);
							progress.resolvedModel = display;
							progress.resolvedModelIdentity = display;
							progress.contextWindow = inferContextWindow(ev.init.model);
							scheduleProgress();
						}
						return;
					}
					if (ev.event === "step_update" && ev.step_update) {
						const su = ev.step_update;
						if (su.step_type === "tool") {
							const toolName = su.tool_name || su.tool_info?.name || "tool";
							if (su.state === "ACTIVE") {
								const summarized = summarizeToolArgs(su.tool_info?.parameters, runCwd);
								activeToolName = toolName;
								activeToolArgs = summarized.args;
								activeToolArgsKey = summarized.argsKey;
								progress.currentTool = toolName;
								progress.currentToolArgs = summarized.args;
								progress.currentToolArgsKey = summarized.argsKey;
								progress.currentToolStartMs = Date.now();
								progress.streaming = undefined;
								scheduleProgress();
							} else if (su.state === "DONE") {
								const summarized = summarizeToolArgs(su.tool_info?.parameters, runCwd);
								progress.toolCount += 1;
								progress.recentTools.push({
									tool: activeToolName ?? toolName,
									args: summarized.args || activeToolArgs,
									argsKey: summarized.argsKey ?? activeToolArgsKey,
									endMs: Date.now(),
								});
								if (progress.recentTools.length > RECENT_TOOLS_LIMIT) {
									progress.recentTools.shift();
								}
								activeToolName = undefined;
								activeToolArgs = "";
								activeToolArgsKey = undefined;
								progress.currentTool = undefined;
								progress.currentToolArgs = undefined;
								progress.currentToolArgsKey = undefined;
								progress.currentToolStartMs = undefined;
								scheduleProgress();
							}
						} else if (su.step_type === "agent_response") {
							if (su.text_delta) {
								recentTextChunks.push(su.text_delta);
							}
							if (su.state === "ACTIVE") {
								progress.streaming = { kind: "text" };
								scheduleProgress();
							} else if (su.state === "DONE") {
								progress.streaming = undefined;
								progress.requests += 1;
								if (su.usage) {
									const inp = su.usage.input_tokens ?? 0;
									const out = su.usage.output_tokens ?? 0;
									const cacheRead = su.usage.cache_read_tokens ?? 0;
									totalInputTokens += inp;
									totalOutputTokens += out;
									totalCacheReadTokens += cacheRead;
									progress.tokens += inp + out;
									progress.contextTokens = inp + cacheRead;
								}
								scheduleProgress();
							}
						}
					}
				},
			});

			conversationId = lastOutcome.conversationId ?? conversationId;
			if (progress.requests === 0 && lastOutcome.numTurns && lastOutcome.numTurns > 0) {
				progress.requests = lastOutcome.numTurns;
			}
			if (totalInputTokens === 0 && totalOutputTokens === 0 && lastOutcome.usage) {
				const inp = lastOutcome.usage.input_tokens ?? 0;
				const out = lastOutcome.usage.output_tokens ?? 0;
				const cacheRead = lastOutcome.usage.cache_read_tokens ?? 0;
				totalInputTokens += inp;
				totalOutputTokens += out;
				totalCacheReadTokens += cacheRead;
				progress.tokens += inp + out;
				progress.contextTokens = inp + cacheRead;
			}
			if (lastOutcome.model) {
				const display = formatResolvedAgyModel(lastOutcome.model);
				progress.resolvedModel = display;
				progress.resolvedModelIdentity = display;
				progress.contextWindow = inferContextWindow(lastOutcome.model);
			}
			if (lastOutcome.exitCode !== 0 || lastOutcome.aborted) {
				break;
			}
		}
	} finally {
		if (runtimeTimer) clearTimeout(runtimeTimer);
		signal?.removeEventListener("abort", onSignalAbort);
		options.launchSignal?.removeEventListener("abort", onLaunchAbort);
		if (progressTimer) {
			clearTimeout(progressTimer);
			progressTimer = null;
		}
	}

	const wasAborted = runtimeExceeded || lastOutcome.aborted || Boolean(signal?.aborted);
	if (wasAborted && !firstTurnNotified) {
		firstTurnNotified = true;
		options.onFirstTurn?.({ kind: "cancelled" });
	}
	const abortReason = wasAborted
		? runtimeExceeded
			? `Subagent exceeded task.maxRuntimeMs (${maxRuntimeMs}ms)`
			: (lastOutcome.abortReason ?? abortReasonOf(signal, "Cancelled"))
		: undefined;
	let exitCode = wasAborted ? 1 : lastOutcome.exitCode;
	let error = wasAborted ? (abortReason ?? lastOutcome.error) : lastOutcome.error;

	let rawOutput = lastOutcome.response;
	if (wasAborted && !rawOutput.trim() && recentTextChunks.length > 0) {
		const snippet = previewLine(recentTextChunks.join("").trim(), 200);
		rawOutput = `[cancelled after ${progress.requests} req, ${progress.tokens} tok — last activity: "${snippet}"]`;
	}

	const structured = finalizeStructuredOutput({
		outputSchema,
		outputSchemaMode: options.outputSchemaMode,
		outputSchemaSource: options.outputSchemaSource,
		exitCode,
		rawOutput,
		parsedFromAgy: lastOutcome.structuredOutput,
		error,
	});
	rawOutput = structured.rawOutput;
	exitCode = structured.exitCode;
	error = structured.error;

	const { content: truncatedOutput, truncated } = truncateTail(rawOutput, {
		maxBytes: MAX_OUTPUT_BYTES,
		maxLines: MAX_OUTPUT_LINES,
	});

	let outputPath: string | undefined;
	let outputMeta: { lineCount: number; charCount: number } | undefined;
	if (options.artifactsDir) {
		const candidatePath = path.join(options.artifactsDir, `${id}.md`);
		try {
			await writeArtifact(candidatePath, rawOutput);
			outputPath = candidatePath;
			outputMeta = {
				lineCount: rawOutput.split("\n").length,
				charCount: rawOutput.length,
			};
		} catch (err) {
			logger.warn("Failed to persist agy subagent output artifact", {
				agentId: id,
				path: candidatePath,
				error: err instanceof Error ? err.message : String(err),
			});
		}
		if (outputPath && structured.structuredOutput && Object.hasOwn(structured.structuredOutput, "data")) {
			const sidecarPath = path.join(options.artifactsDir, `${id}.json`);
			try {
				const serialized = JSON.stringify(structured.structuredOutput.data, null, 2);
				if (serialized !== undefined) await writeArtifact(sidecarPath, `${serialized}\n`);
			} catch (err) {
				logger.warn("Failed to persist agy subagent structured output sidecar", {
					agentId: id,
					path: sidecarPath,
					error: err instanceof Error ? err.message : String(err),
				});
			}
		}
	}

	progress.currentTool = undefined;
	progress.currentToolArgs = undefined;
	progress.currentToolArgsKey = undefined;
	progress.currentToolStartMs = undefined;
	progress.streaming = undefined;
	progress.status = wasAborted ? "aborted" : exitCode === 0 ? "completed" : "failed";
	scheduleProgress(true);

	emitSubagentFrame(options.eventBus, options.subagentEventBus, TASK_SUBAGENT_LIFECYCLE_CHANNEL, {
		id,
		agent: agent.name,
		parentToolCallId: options.parentToolCallId,
		detached: options.detached,
		agentSource: agent.source,
		description: progress.description,
		status: progress.status,
		index,
	});

	if (outputPath) {
		AgentRegistry.global().setHistory(id, { outputPath });
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
		truncated: Boolean(truncated),
		...(structured.structuredOutput ? { structuredOutput: structured.structuredOutput } : {}),
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
			input: totalInputTokens,
			output: totalOutputTokens,
			cacheRead: totalCacheReadTokens,
			cacheWrite: 0,
			totalTokens: totalInputTokens + totalOutputTokens + totalCacheReadTokens,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		outputPath,
		outputMeta,
	};
}
