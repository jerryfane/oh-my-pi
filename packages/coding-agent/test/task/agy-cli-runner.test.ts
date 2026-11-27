import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { Settings } from "../../src/config/settings";
import { resolveAgySubagentConfig } from "../../src/task/agy-cli-runner";
import { type ExecutorOptions, runSubprocess } from "../../src/task/executor";
import type { AgentProgress } from "../../src/task/task-details";
import type { AgentDefinition, SubagentLifecyclePayload, SubagentProgressPayload } from "../../src/task/types";
import { TASK_SUBAGENT_LIFECYCLE_CHANNEL, TASK_SUBAGENT_PROGRESS_CHANNEL } from "../../src/task/types";
import { EventBus } from "../../src/utils/event-bus";

const tempDirs: string[] = [];

async function makeTempDir(prefix: string): Promise<string> {
	const dir = await fs.mkdtemp(path.join(os.tmpdir(), prefix));
	tempDirs.push(dir);
	return dir;
}

async function createFakeAgyBinary(dir: string, scriptBody: string): Promise<string> {
	const binPath = path.join(dir, "fake-agy");
	const full = `#!/usr/bin/env bun\n${scriptBody}\n`;
	await fs.writeFile(binPath, full, { mode: 0o755 });
	return binPath;
}

function makeAgent(overrides: Partial<AgentDefinition> = {}): AgentDefinition {
	return {
		name: "task",
		description: "General worker",
		systemPrompt: "Be concise.",
		source: "bundled",
		...overrides,
	};
}

function makeSettings(agyBinary: string, agySubagents: Record<string, string | boolean>): Settings {
	return Settings.isolated({
		"task.agyBinary": agyBinary,
		"task.agySubagents": agySubagents,
	});
}

afterEach(async () => {
	while (tempDirs.length > 0) {
		const dir = tempDirs.pop()!;
		await fs.rm(dir, { recursive: true, force: true });
	}
});

describe("agy CLI subagent runner", () => {
	it("resolves exact agent overrides, wildcard defaults, opt-outs, effort suffixes, and OMP-only fallbacks", async () => {
		const dir = await makeTempDir("agy-runner-resolve-");
		const bin = await createFakeAgyBinary(dir, "process.exit(0);");

		const settings = makeSettings(bin, {
			"*": "gemini-4-neon",
			scout: "gemini-3.8-flash-low",
			reviewer: "off",
		});

		const base = (overrides: Partial<ExecutorOptions> = {}): ExecutorOptions => ({
			cwd: dir,
			agent: makeAgent(),
			task: "hello",
			index: 0,
			id: "Worker-1",
			settings,
			...overrides,
		});

		// Wildcard + coarse effort suffix applied when model has no suffix
		expect(resolveAgySubagentConfig(base({ effort: "med" }))).toEqual({
			binary: bin,
			model: "gemini-4-neon-medium",
		});

		// Exact agent overrides wildcard; existing effort suffix is preserved
		expect(resolveAgySubagentConfig(base({ agent: makeAgent({ name: "scout" }), effort: "hi" }))).toEqual({
			binary: bin,
			model: "gemini-3.8-flash-low",
		});

		// Explicit "off" opts an agent out of wildcard
		expect(resolveAgySubagentConfig(base({ agent: makeAgent({ name: "reviewer" }) }))).toBeUndefined();

		// OMP-only contracts fall back to native OMP
		expect(resolveAgySubagentConfig(base({ restrictToolNames: true }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ topicBoard: "board" }))).toBeUndefined();
		expect(
			resolveAgySubagentConfig(
				base({
					perCallModelGrants: { settings, agentName: "task" },
				}),
			),
		).toBeUndefined();

		// Missing binary falls back safely
		const missingSettings = makeSettings(path.join(dir, "does-not-exist"), { "*": "gemini-4-neon-medium" });
		expect(resolveAgySubagentConfig(base({ settings: missingSettings }))).toBeUndefined();
	});

	it("runs a real subprocess through runSubprocess, streams HUD progress/lifecycle, tracks tokens, and writes artifacts", async () => {
		const dir = await makeTempDir("agy-runner-stream-");
		const artifactsDir = path.join(dir, "artifacts");
		await fs.mkdir(artifactsDir, { recursive: true });
		const argvDump = path.join(dir, "argv.json");

		const bin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
fs.writeFileSync(${JSON.stringify(argvDump)}, JSON.stringify(process.argv.slice(2)));
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-123", init: { model: "gemini-4-neon-medium", cwd: process.cwd(), permission_mode: "always-proceed" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", parameters: { Path: "/tmp/demo.ts" } } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 1, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", parameters: { Path: "/tmp/demo.ts" }, output: "10 lines" } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "ACTIVE", step_type: "agent_response", text_delta: "Found " } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "DONE", step_type: "agent_response", text_delta: "target line\\n", usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 9000, total_tokens: 1280 } } });
emit({ event: "result", result: { conversation_id: "conv-123", status: "SUCCESS", response: "Found target line\\n", duration_seconds: 0.4, num_turns: 1, usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 9000, total_tokens: 1280 } } });
`,
		);

		const eventBus = new EventBus();
		const lifecycles: SubagentLifecyclePayload[] = [];
		const progressFrames: SubagentProgressPayload[] = [];
		const progressCallbacks: AgentProgress[] = [];
		eventBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, data => lifecycles.push(data as SubagentLifecyclePayload));
		eventBus.on(TASK_SUBAGENT_PROGRESS_CHANNEL, data => progressFrames.push(data as SubagentProgressPayload));

		const scout = makeAgent({ name: "scout", tools: ["read", "grep", "find"] });
		const result = await runSubprocess({
			cwd: dir,
			agent: scout,
			task: "--check-leading-dash find target",
			context: "Shared repo context",
			description: "Scout demo file",
			index: 0,
			id: "ScoutDemo",
			detached: true,
			artifactsDir,
			eventBus,
			settings: makeSettings(bin, { scout: "gemini-4-neon-medium" }),
			onProgress: p => progressCallbacks.push(p),
		});

		expect(result.exitCode).toBe(0);
		expect(result.output).toBe("Found target line\n");
		expect(result.resolvedModel).toBe("agy-cli/gemini-4-neon-medium");
		expect(result.requests).toBe(1);
		// Cumulative tokens exclude cache_read_tokens; contextTokens includes input + cache_read
		expect(result.tokens).toBe(1280);
		expect(result.contextTokens).toBe(10200);
		expect(result.contextWindow).toBe(1_000_000);
		expect(result.outputPath).toBe(path.join(artifactsDir, "ScoutDemo.md"));
		expect(await fs.readFile(result.outputPath!, "utf8")).toBe("Found target line\n");

		// Verify lifecycle & HUD progress events
		expect(lifecycles.map(l => l.status)).toEqual(["started", "completed"]);
		expect(lifecycles[0]?.detached).toBe(true);
		expect(progressCallbacks.at(-1)?.status).toBe("completed");
		expect(progressCallbacks.at(-1)?.toolCount).toBe(1);
		expect(progressCallbacks.at(-1)?.recentTools[0]?.tool).toBe("read_file");
		expect(progressFrames.length).toBeGreaterThanOrEqual(2);

		// Verify CLI flags, leading-dash safety (`--print=...`), shared context, and READ-ONLY guidance
		const argv = JSON.parse(await fs.readFile(argvDump, "utf8")) as string[];
		expect(argv).toContain("--output-format");
		expect(argv).toContain("stream-json");
		expect(argv).toContain("--dangerously-skip-permissions");
		expect(argv).toContain("--model");
		expect(argv).toContain("gemini-4-neon-medium");
		const printArg = argv.find(a => a.startsWith("--print="));
		expect(printArg).toBeDefined();
		expect(printArg).toContain("READ-ONLY");
		expect(printArg).toContain("Shared repo context");
		expect(printArg).toContain("--check-leading-dash find target");
	});

	it("enforces structured outputSchema via --json-schema, writes sidecar JSON, and rejects invalid strict output", async () => {
		const dir = await makeTempDir("agy-runner-schema-");
		const artifactsDir = path.join(dir, "artifacts");
		await fs.mkdir(artifactsDir, { recursive: true });
		const schemaCapture = path.join(dir, "schema.txt");

		const validBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
const idx = process.argv.indexOf("--json-schema");
if (idx >= 0) fs.writeFileSync(${JSON.stringify(schemaCapture)}, process.argv[idx + 1] ?? "");
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-s1", init: { model: "gemini-4-neon-low" } });
emit({ event: "result", result: { conversation_id: "conv-s1", status: "SUCCESS", response: '{"summary":"all good"}', structured_output: { summary: "all good" }, num_turns: 1, usage: { input_tokens: 100, output_tokens: 20 } } });
`,
		);

		const jtdSchema = { properties: { summary: { type: "string" } } };
		const ok = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "scout", tools: ["read"] }),
			task: "summarize",
			index: 0,
			id: "SchemaOk",
			artifactsDir,
			outputSchema: jtdSchema,
			outputSchemaMode: "strict",
			outputSchemaSource: "agent",
			settings: makeSettings(validBin, { scout: "gemini-4-neon-low" }),
		});

		expect(ok.exitCode).toBe(0);
		expect(ok.structuredOutput).toEqual({
			source: "agent",
			mode: "strict",
			status: "valid",
			data: { summary: "all good" },
		});
		const passedSchema = JSON.parse(await fs.readFile(schemaCapture, "utf8"));
		expect(passedSchema.type).toBe("object");
		expect(passedSchema.required).toEqual(["summary"]);
		expect(JSON.parse(await fs.readFile(path.join(artifactsDir, "SchemaOk.json"), "utf8"))).toEqual({
			summary: "all good",
		});

		// Strict invalid output fails with exitCode 1
		const badBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-s2", init: { model: "gemini-4-neon-low" } });
emit({ event: "result", result: { conversation_id: "conv-s2", status: "SUCCESS", response: '{"wrong":123}', structured_output: { wrong: 123 }, num_turns: 1 } });
`,
		);
		const bad = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "scout", tools: ["read"] }),
			task: "summarize",
			index: 0,
			id: "SchemaBad",
			outputSchema: jtdSchema,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			settings: makeSettings(badBin, { scout: "gemini-4-neon-low" }),
		});
		expect(bad.exitCode).toBe(1);
		expect(bad.structuredOutput?.status).toBe("invalid");
	});

	it("chains multi-step tasks via --conversation and spills oversized prompts safely", async () => {
		const dir = await makeTempDir("agy-runner-steps-");
		const logPath = path.join(dir, "invocations.jsonl");

		const bin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const printArg = argv.find(a => a.startsWith("--print=")) ?? "";
fs.appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ conv, printLen: printArg.length, printArg: printArg.slice(0, 140) }) + "\\n");
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
const cid = conv ?? "conv-step-1";
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: conv ? "step-2-done" : "step-1-done", num_turns: 1 } });
`,
		);

		const hugePrompt = `Step 1: ${"x".repeat(110_000)}`;
		const result = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: hugePrompt,
			steps: [hugePrompt, "Step 2: wrap up"],
			index: 0,
			id: "MultiStep",
			artifactsDir: dir,
			settings: makeSettings(bin, { "*": "gemini-4-neon-medium" }),
		});

		expect(result.exitCode).toBe(0);
		expect(result.output).toBe("step-2-done");
		const calls = (await fs.readFile(logPath, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as { conv: string | null; printLen: number; printArg: string });
		expect(calls.length).toBe(2);
		expect(calls[0]?.conv).toBeNull();
		// Oversized step 1 spilled to a temporary file rather than overflowing argv
		expect(calls[0]?.printLen).toBeLessThan(1024);
		expect(calls[0]?.printArg).toContain("agy-prompt-0");
		expect(calls[1]?.conv).toBe("conv-step-1");
		// Spill prompt file cleaned up
		const leftovers = (await fs.readdir(dir)).filter(f => f.includes("agy-prompt-"));
		expect(leftovers).toEqual([]);
	});

	it("aborts cleanly on signal or maxRuntimeMs timeout and surfaces error frames", async () => {
		const dir = await makeTempDir("agy-runner-abort-");
		const slowBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-slow", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-slow", step_index: 1, state: "ACTIVE", step_type: "agent_response", text_delta: "partial progress here" } });
setTimeout(() => {}, 30_000);
`,
		);

		const aborted = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "long job",
			index: 0,
			id: "TimeoutJob",
			maxRuntimeMs: 300,
			settings: makeSettings(slowBin, { "*": "gemini-4-neon-low" }),
		});
		expect(aborted.exitCode).toBe(1);
		expect(aborted.aborted).toBe(true);
		expect(aborted.abortReason).toContain("maxRuntimeMs");
		expect(aborted.output).toContain("partial progress here");

		const errBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "result", result: { conversation_id: "conv-err", status: "ERROR", response: "", error: "boom from agy", num_turns: 0 } });
process.exit(1);
`,
		);
		const failed = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "failing job",
			index: 0,
			id: "ErrorJob",
			settings: makeSettings(errBin, { "*": "gemini-4-neon-low" }),
		});
		expect(failed.exitCode).toBe(1);
		expect(failed.aborted).toBeUndefined();
		expect(failed.error).toContain("boom from agy");
	});
});
