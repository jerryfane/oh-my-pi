import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { Settings } from "../../src/config/settings";
import { resolveAgySubagentConfig } from "../../src/task/agy-cli-runner";
import { type ExecutorOptions, runSubprocess, type SubagentToolResult } from "../../src/task/executor";
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

async function createFakeAgyBinary(dir: string, scriptBody: string, name = "fake-agy"): Promise<string> {
	const binPath = path.join(dir, name);
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

function makeSettings(
	agyBinary: string,
	agySubagents: Record<string, string | boolean>,
	extra: Record<string, unknown> = {},
): Settings {
	return Settings.isolated({
		"task.agyBinary": agyBinary,
		"task.agySubagents": agySubagents,
		...extra,
	});
}

afterEach(async () => {
	while (tempDirs.length > 0) {
		const dir = tempDirs.pop()!;
		await fs.rm(dir, { recursive: true, force: true });
	}
});

describe("agy CLI subagent runner", () => {
	it("validates settings and enforces routing boundaries (read-only, offline, keepAlive, topic seed, OMP-only)", async () => {
		const dir = await makeTempDir("agy-runner-resolve-");
		const bin = await createFakeAgyBinary(dir, "process.exit(0);");

		// Invalid settings values are rejected by the settings validator
		expect(() => Settings.isolated({ "task.agySubagents": { task: 123 } })).toThrow(/task\.agySubagents\.task/);
		expect(() => Settings.isolated({ "task.agySubagents": { task: "   " } })).toThrow(/task\.agySubagents\.task/);

		const settings = makeSettings(bin, {
			"*": "gemini-4-neon-medium",
			reviewer: "gemini-4-neon-high",
			special: "off",
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

		// Wildcard matches writable task verbatim (no guessed effort mutation)
		expect(resolveAgySubagentConfig(base({ effort: "lo" }))).toEqual({
			binary: bin,
			model: "gemini-4-neon-medium",
		});

		// Exact agent overrides wildcard (reviewer has bash so it is not read-only)
		expect(
			resolveAgySubagentConfig(base({ agent: makeAgent({ name: "reviewer", tools: ["read", "bash"] }) })),
		).toEqual({
			binary: bin,
			model: "gemini-4-neon-high",
		});

		// Explicit "off" opts out of wildcard
		expect(resolveAgySubagentConfig(base({ agent: makeAgent({ name: "special" }) }))).toBeUndefined();

		// Read-only agents stay on native OMP so tool/filesystem restrictions remain enforced
		expect(
			resolveAgySubagentConfig(base({ agent: makeAgent({ name: "scout", tools: ["read", "grep", "find"] }) })),
		).toBeUndefined();

		// Offline mode pins dispatch to parent provider on native OMP
		const offlineSettings = makeSettings(bin, { "*": "gemini-4-neon-medium" }, { "models.offline": true });
		expect(resolveAgySubagentConfig(base({ settings: offlineSettings }))).toBeUndefined();

		// Seeded/resumable topic lifecycles and OMP-only contracts stay on native OMP
		expect(resolveAgySubagentConfig(base({ keepAlive: true }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ topicBoard: "board" }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ launchSignal: new AbortController().signal }))).toBeUndefined();
		expect(
			resolveAgySubagentConfig(base({ launchPreamble: { text: "seed", fit: () => ({ text: "seed", tokens: 1 }) } })),
		).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ onFirstTurn: () => {} }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ restrictToolNames: true }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ perCallModelGrants: { settings, agentName: "task" } }))).toBeUndefined();

		// Missing binary falls back safely
		const missingSettings = makeSettings(path.join(dir, "does-not-exist"), { "*": "gemini-4-neon-medium" });
		expect(resolveAgySubagentConfig(base({ settings: missingSettings }))).toBeUndefined();
	});

	it("streams HUD progress newest-first, forwards onToolResult, tracks tokens, and clears stale JSON sidecars", async () => {
		const dir = await makeTempDir("agy-runner-stream-");
		const artifactsDir = path.join(dir, "artifacts");
		await fs.mkdir(artifactsDir, { recursive: true });
		const argvDump = path.join(dir, "argv.json");
		// Seed a stale JSON sidecar from an earlier run to prove plain replacement clears it
		const staleSidecar = path.join(artifactsDir, "WorkerDemo.json");
		await fs.writeFile(staleSidecar, '{"stale":true}\n');

		const bin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
fs.writeFileSync(${JSON.stringify(argvDump)}, JSON.stringify(process.argv.slice(2)));
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-123", init: { model: "gemini-4-neon-medium", cwd: process.cwd(), permission_mode: "always-proceed" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", parameters: { Path: "/tmp/demo.ts" } } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 1, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", parameters: { Path: "/tmp/demo.ts" }, output: "10 lines read" } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "ACTIVE", step_type: "agent_response", text_delta: "line 1\\nline 2\\n" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "DONE", step_type: "agent_response", text_delta: "line 3\\n", usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 9000, total_tokens: 1280 } } });
emit({ event: "result", result: { conversation_id: "conv-123", status: "SUCCESS", response: "line 1\\nline 2\\nline 3\\n", duration_seconds: 0.4, num_turns: 1, usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 9000, total_tokens: 1280 } } });
`,
		);

		const eventBus = new EventBus();
		const lifecycles: SubagentLifecyclePayload[] = [];
		const progressFrames: SubagentProgressPayload[] = [];
		const progressCallbacks: AgentProgress[] = [];
		const toolResults: SubagentToolResult[] = [];
		eventBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, data => lifecycles.push(data as SubagentLifecyclePayload));
		eventBus.on(TASK_SUBAGENT_PROGRESS_CHANNEL, data => progressFrames.push(data as SubagentProgressPayload));

		const result = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "task" }),
			task: "--check-leading-dash find target",
			context: "Shared repo context",
			description: "Run demo task",
			index: 0,
			id: "WorkerDemo",
			detached: true,
			artifactsDir,
			eventBus,
			settings: makeSettings(bin, { task: "gemini-4-neon-medium" }),
			onProgress: p => progressCallbacks.push(p),
			onToolResult: r => toolResults.push(r),
		});

		expect(result.exitCode).toBe(0);
		expect(result.output).toBe("line 1\nline 2\nline 3\n");
		expect(result.resolvedModel).toBe("agy-cli/gemini-4-neon-medium");
		expect(result.requests).toBe(1);
		expect(result.tokens).toBe(1280);
		expect(result.contextTokens).toBe(10200);
		expect(result.contextWindow).toBeUndefined();
		expect(result.outputPath).toBe(path.join(artifactsDir, "WorkerDemo.md"));
		expect(await fs.readFile(result.outputPath!, "utf8")).toBe("line 1\nline 2\nline 3\n");
		// Stale JSON sidecar cleared when plain Markdown replaced it
		expect(await Bun.file(staleSidecar).exists()).toBe(false);

		// Lifecycle, newest-first recentOutput, and onToolResult callback
		expect(lifecycles.map(l => l.status)).toEqual(["started", "completed"]);
		expect(lifecycles[0]?.detached).toBe(true);
		expect(progressCallbacks.at(-1)?.status).toBe("completed");
		expect(progressCallbacks.at(-1)?.toolCount).toBe(1);
		expect(progressCallbacks.at(-1)?.recentTools[0]?.tool).toBe("read_file");
		expect(progressCallbacks.at(-1)?.recentOutput).toEqual(["line 3", "line 2", "line 1"]);
		expect(toolResults).toEqual([{ tool: "read_file", text: "10 lines read", isError: false }]);
		expect(progressFrames.length).toBeGreaterThanOrEqual(2);

		const argv = JSON.parse(await fs.readFile(argvDump, "utf8")) as string[];
		expect(argv).toContain("--output-format");
		expect(argv).toContain("stream-json");
		expect(argv).toContain("--dangerously-skip-permissions");
		expect(argv).toContain("--model");
		expect(argv).toContain("gemini-4-neon-medium");
		const printArg = argv.find(a => a.startsWith("--print="));
		expect(printArg).toBeDefined();
		expect(printArg).toContain("Shared repo context");
		expect(printArg).toContain("--check-leading-dash find target");
	});

	it("validates constrained, unconstrained (true), and strict-invalid structured output schemas", async () => {
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
			"fake-agy-valid",
		);

		const jtdSchema = { properties: { summary: { type: "string" } } };
		const ok = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "reviewer", tools: ["read", "bash"] }),
			task: "summarize",
			index: 0,
			id: "SchemaOk",
			artifactsDir,
			outputSchema: jtdSchema,
			outputSchemaMode: "strict",
			outputSchemaSource: "agent",
			settings: makeSettings(validBin, { reviewer: "gemini-4-neon-low" }),
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

		// Unconstrained JSON Schema `true` is valid and publishes structured data + sidecar
		const unconstrained = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "task" }),
			task: "summarize",
			index: 0,
			id: "SchemaTrue",
			artifactsDir,
			outputSchema: true,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			settings: makeSettings(validBin, { task: "gemini-4-neon-low" }),
		});
		expect(unconstrained.exitCode).toBe(0);
		expect(unconstrained.structuredOutput).toEqual({
			source: "caller",
			mode: "strict",
			status: "valid",
			data: { summary: "all good" },
		});
		expect(JSON.parse(await fs.readFile(path.join(artifactsDir, "SchemaTrue.json"), "utf8"))).toEqual({
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
			"fake-agy-bad",
		);
		const bad = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "task" }),
			task: "summarize",
			index: 0,
			id: "SchemaBad",
			outputSchema: jtdSchema,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			settings: makeSettings(badBin, { task: "gemini-4-neon-low" }),
		});
		expect(bad.exitCode).toBe(1);
		expect(bad.structuredOutput?.status).toBe("invalid");
	});

	it("accumulates multi-step usage per step, spills oversized prompts, and settles cleanly on spill failure or missing result", async () => {
		const dir = await makeTempDir("agy-runner-steps-");
		const logPath = path.join(dir, "invocations.jsonl");

		// Step 1 reports usage on step_update DONE; Step 2 reports usage ONLY on result frame
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
if (!conv) {
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 1, state: "DONE", step_type: "agent_response", text_delta: "step-1-done", usage: { input_tokens: 100, output_tokens: 10, cache_read_tokens: 500 } } });
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "step-1-done", num_turns: 1, usage: { input_tokens: 100, output_tokens: 10, cache_read_tokens: 500 } } });
} else {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "step-2-done", num_turns: 2, usage: { input_tokens: 200, output_tokens: 20, cache_read_tokens: 600 } } });
}
`,
			"fake-agy-steps",
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
		// Exact per-step reconciliation: 1+2=3 requests, 110+220=330 tokens, 500+600=1100 cacheRead
		expect(result.requests).toBe(3);
		expect(result.tokens).toBe(330);
		expect(result.usage?.cacheRead).toBe(1100);

		const calls = (await fs.readFile(logPath, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as { conv: string | null; printLen: number; printArg: string });
		expect(calls.length).toBe(2);
		expect(calls[0]?.conv).toBeNull();
		expect(calls[0]?.printLen).toBeLessThan(1024);
		expect(calls[1]?.conv).toBe("conv-step-1");
		const leftovers = (await fs.readdir(dir)).filter(f => f.includes("agy-prompt-"));
		expect(leftovers).toEqual([]);

		// Spill write failure emits terminal failed lifecycle instead of rejecting unhandled
		const spillBus = new EventBus();
		const spillLifecycles: SubagentLifecyclePayload[] = [];
		spillBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, d => spillLifecycles.push(d as SubagentLifecyclePayload));
		const spillFail = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: hugePrompt,
			index: 0,
			id: "SpillFail",
			artifactsDir: path.join(dir, "missing-nonexistent-dir"),
			eventBus: spillBus,
			settings: makeSettings(bin, { "*": "gemini-4-neon-medium" }),
		});
		expect(spillFail.exitCode).toBe(1);
		expect(spillFail.error).toBeDefined();
		expect(spillLifecycles.map(l => l.status)).toEqual(["started", "failed"]);

		// Zero-exit process without a terminal result frame fails and stops multi-step execution
		const truncatedBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-trunc", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-trunc", step_index: 1, state: "ACTIVE", step_type: "agent_response", text_delta: "halfway output" } });
process.exit(0);
`,
			"fake-agy-truncated",
		);
		const truncatedRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "step 1",
			steps: ["step 1", "step 2 should not run"],
			index: 0,
			id: "NoResultFrame",
			settings: makeSettings(truncatedBin, { "*": "gemini-4-neon-low" }),
		});
		expect(truncatedRes.exitCode).toBe(1);
		expect(truncatedRes.error).toContain("terminal result");
		expect(truncatedRes.output).toBe("halfway output");
	});

	it("kills the entire process tree on timeout without hanging on descendants and surfaces error frames", async () => {
		const dir = await makeTempDir("agy-runner-abort-");
		const sentinelPath = path.join(dir, "descendant-survived.txt");

		// Spawn a descendant that inherits stdout and tries to write a sentinel after 600ms
		const treeBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-tree", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-tree", step_index: 1, state: "ACTIVE", step_type: "agent_response", text_delta: "partial progress here" } });
Bun.spawn(["sh", "-c", "sleep 0.6 && echo alive > " + ${JSON.stringify(sentinelPath)}], { stdout: "inherit", stderr: "inherit" });
setTimeout(() => {}, 30_000);
`,
			"fake-agy-tree",
		);

		const t0 = Date.now();
		const aborted = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "long job with descendant",
			index: 0,
			id: "TimeoutTreeJob",
			maxRuntimeMs: 180,
			settings: makeSettings(treeBin, { "*": "gemini-4-neon-low" }),
		});
		const elapsed = Date.now() - t0;
		expect(aborted.exitCode).toBe(1);
		expect(aborted.aborted).toBe(true);
		expect(aborted.abortReason).toContain("maxRuntimeMs");
		expect(aborted.output).toContain("partial progress here");
		expect(elapsed).toBeLessThan(4_000);

		// Wait past the descendant's sleep window and verify it was killed with the tree
		await Bun.sleep(700);
		expect(await Bun.file(sentinelPath).exists()).toBe(false);

		const errBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "result", result: { conversation_id: "conv-err", status: "ERROR", response: "", error: "boom from agy", num_turns: 0 } });
process.exit(1);
`,
			"fake-agy-err",
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
