import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { Settings } from "../../src/config/settings";
import { CostGate } from "../../src/session/cost-gate";
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

async function createFakeAgyShell(dir: string, scriptBody: string, name = "fake-agy-sh"): Promise<string> {
	const binPath = path.join(dir, name);
	const full = `#!/bin/sh\n${scriptBody}\n`;
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

		// Spend-capped sessions stay native so CostGate enforces session.maxCost; uncapped CostGate still routes to agy
		const cappedSettings = makeSettings(bin, { "*": "gemini-4-neon-medium" }, { "session.maxCost": 1.0 });
		const cappedGate = new CostGate(cappedSettings, 1.25, () => {});
		expect(resolveAgySubagentConfig(base({ settings: cappedSettings, costGate: cappedGate }))).toBeUndefined();
		expect(resolveAgySubagentConfig(base({ costGate: cappedGate }))).toBeUndefined();
		const uncappedGate = new CostGate(settings, 0, () => {});
		expect(resolveAgySubagentConfig(base({ costGate: uncappedGate }))).toEqual({
			binary: bin,
			model: "gemini-4-neon-medium",
		});

		// Relative task.agyBinary is canonicalized through symlinks (`link/../agy` -> `real/agy`, not lexical `./agy`)
		const symBase = await makeTempDir("agy-runner-sym-");
		const realDir = path.join(symBase, "real");
		const nestedDir = path.join(realDir, "nested");
		await fs.mkdir(nestedDir, { recursive: true });
		const expectedSymBin = await createFakeAgyShell(realDir, "exit 0", "agy");
		await createFakeAgyShell(symBase, "exit 1", "agy");
		const linkPath = path.join(symBase, "link");
		await fs.symlink(nestedDir, linkPath);
		// Do not use path.join(..., "..", ...), which lexically collapses `link/..` before config lookup
		const relativeSymBin = `${path.relative(process.cwd(), linkPath)}${path.sep}..${path.sep}agy`;
		const relativeSettings = makeSettings(relativeSymBin, { task: "gemini-4-neon-low" });
		const otherCwd = await makeTempDir("agy-runner-other-cwd-");
		expect(resolveAgySubagentConfig(base({ cwd: otherCwd, settings: relativeSettings }))).toEqual({
			binary: await fs.realpath(expectedSymBin),
			model: "gemini-4-neon-low",
		});

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
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "ACTIVE", step_type: "tool", tool_name: "list_directory", tool_info: { name: "list_directory", parameters: { Path: "/tmp" } } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 1, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", output: "10 lines read" } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 2, state: "DONE", step_type: "tool", tool_name: "list_directory", tool_info: { name: "list_directory", output: "demo.ts" } } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 3, state: "ACTIVE", step_type: "agent_response", text_delta: "line 1\\nline 2\\n" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-123", step_index: 3, state: "DONE", step_type: "agent_response", text_delta: "line 3\\n", usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 9000, total_tokens: 1280 } } });
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

		// Lifecycle, newest-first recentOutput, and interleaved onToolResult attribution
		expect(lifecycles.map(l => l.status)).toEqual(["started", "completed"]);
		expect(lifecycles[0]?.detached).toBe(true);
		expect(progressCallbacks.at(-1)?.status).toBe("completed");
		expect(progressCallbacks.at(-1)?.toolCount).toBe(2);
		expect(progressCallbacks.at(-1)?.recentTools.map(t => ({ tool: t.tool, args: t.args }))).toEqual([
			{ tool: "read_file", args: "/tmp/demo.ts" },
			{ tool: "list_directory", args: "/tmp" },
		]);
		expect(progressCallbacks.at(-1)?.recentOutput).toEqual(["line 3", "line 2", "line 1"]);
		expect(toolResults).toEqual([
			{ tool: "read_file", text: "10 lines read", isError: false },
			{ tool: "list_directory", text: "demo.ts", isError: false },
		]);
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

	it("validates constrained, unconstrained (true / {}), non-object, and strict-invalid structured output schemas", async () => {
		const dir = await makeTempDir("agy-runner-schema-");
		const artifactsDir = path.join(dir, "artifacts");
		await fs.mkdir(artifactsDir, { recursive: true });
		const schemaCapture = path.join(dir, "schema.txt");

		// Emits JSON ONLY when requested via --json-schema (which must have root type:"object", matching real agy)
		// or via an explicit <output_format> JSON instruction in the prompt; otherwise emits plain prose.
		const validBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
const argv = process.argv.slice(2);
const idx = argv.indexOf("--json-schema");
const printArg = argv.find(a => a.startsWith("--print=")) ?? "";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
if (idx >= 0) {
	const rawSchema = argv[idx + 1] ?? "";
	const parsed = JSON.parse(rawSchema);
	if (parsed?.type !== "object") {
		process.stderr.write('Error: invalid --json-schema: schema root must specify "type": "object"\\n');
		process.exit(1);
	}
	fs.writeFileSync(${JSON.stringify(schemaCapture)}, rawSchema);
	emit({ event: "init", conversation_id: "conv-s1", init: { model: "gemini-4-neon-low" } });
	emit({ event: "result", result: { conversation_id: "conv-s1", status: "SUCCESS", response: '{"summary":"all good"}', structured_output: { summary: "all good" }, num_turns: 1, usage: { input_tokens: 100, output_tokens: 20 } } });
} else if (printArg.includes("<output_format>") && printArg.includes("raw valid JSON")) {
	const payload = printArg.includes('"type":"array"') ? '["a","b"]' : '{"summary":"all good"}';
	emit({ event: "init", conversation_id: "conv-s1", init: { model: "gemini-4-neon-low" } });
	emit({ event: "result", result: { conversation_id: "conv-s1", status: "SUCCESS", response: payload, num_turns: 1, usage: { input_tokens: 100, output_tokens: 20 } } });
} else {
	emit({ event: "init", conversation_id: "conv-s1", init: { model: "gemini-4-neon-low" } });
	emit({ event: "result", result: { conversation_id: "conv-s1", status: "SUCCESS", response: "Plain prose response.", num_turns: 1 } });
}
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

		// Unconstrained JSON Schema `true` and `{}` both request raw JSON and publish structured data + sidecar
		for (const [label, schemaVal] of [
			["SchemaTrue", true],
			["SchemaEmptyObj", {}],
		] as const) {
			const unconstrained = await runSubprocess({
				cwd: dir,
				agent: makeAgent({ name: "task" }),
				task: "summarize",
				index: 0,
				id: label,
				artifactsDir,
				outputSchema: schemaVal,
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
			expect(JSON.parse(await fs.readFile(path.join(artifactsDir, `${label}.json`), "utf8"))).toEqual({
				summary: "all good",
			});
		}

		// Valid non-object JSON Schema (array) requests raw JSON via prompt instead of failing agy --json-schema
		const arrayRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent({ name: "task" }),
			task: "list items",
			index: 0,
			id: "SchemaArray",
			artifactsDir,
			outputSchema: { type: "array", items: { type: "string" } },
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			settings: makeSettings(validBin, { task: "gemini-4-neon-low" }),
		});
		expect(arrayRes.exitCode).toBe(0);
		expect(arrayRes.structuredOutput).toEqual({
			source: "caller",
			mode: "strict",
			status: "valid",
			data: ["a", "b"],
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
	}, 30_000);

	it("accumulates multi-step usage per step, spills oversized prompts, and settles cleanly on spill failure or missing result", async () => {
		const dir = await makeTempDir("agy-runner-steps-");
		const logPath = path.join(dir, "invocations.jsonl");

		// Step 1 reports partial step_update usage (1st DONE has usage, 2nd DONE omits usage) + aggregate on result;
		// Step 2 reports usage ONLY on result frame.
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
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 1, state: "DONE", step_type: "agent_response", text_delta: "part-1", usage: { input_tokens: 100, output_tokens: 10, cache_read_tokens: 40 } } });
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 2, state: "DONE", step_type: "agent_response", text_delta: "part-2" } });
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "step-1-done", num_turns: 2, usage: { input_tokens: 300, output_tokens: 30, cache_read_tokens: 90 } } });
} else {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "step-2-done", num_turns: 2, usage: { input_tokens: 200, output_tokens: 20, cache_read_tokens: 600 } } });
}
`,
			"fake-agy-steps",
		);

		// 90,000-byte multiline prompt (< 96 KiB raw, > 128 KiB JSON-encoded for Linux subreaper) must spill instead of E2BIG
		const multilinePrompt = "a\n".repeat(45_000);
		const result = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: multilinePrompt,
			steps: [multilinePrompt, "Step 2: wrap up"],
			index: 0,
			id: "MultiStep",
			artifactsDir: dir,
			settings: makeSettings(bin, { "*": "gemini-4-neon-medium" }),
		});

		expect(result.exitCode).toBe(0);
		expect(result.output).toBe("step-2-done");
		// Authoritative terminal totals per step: 2+2=4 requests, 330+220=550 tokens, 90+600=690 cacheRead
		expect(result.requests).toBe(4);
		expect(result.tokens).toBe(550);
		expect(result.usage?.cacheRead).toBe(690);

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

		// Single-step partial step_update usage preserves latest per-request contextTokens separately from aggregate usage
		const singlePartial = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "short single step",
			index: 0,
			id: "PartialUsageSingle",
			settings: makeSettings(bin, { "*": "gemini-4-neon-medium" }),
		});
		expect(singlePartial.exitCode).toBe(0);
		expect(singlePartial.requests).toBe(2);
		expect(singlePartial.tokens).toBe(330);
		expect(singlePartial.usage?.cacheRead).toBe(90);
		expect(singlePartial.contextTokens).toBe(140);

		// Spill write failure emits terminal failed lifecycle instead of rejecting unhandled
		const spillBus = new EventBus();
		const spillLifecycles: SubagentLifecyclePayload[] = [];
		spillBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, d => spillLifecycles.push(d as SubagentLifecyclePayload));
		const spillFail = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: multilinePrompt,
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
	}, 30_000);

	it("hard-kills SIGTERM-ignoring descendants at maxRuntimeMs, settles promptly when a helper lingers after SUCCESS, and surfaces error frames", async () => {
		const dir = await makeTempDir("agy-runner-abort-");
		const sentinelPath = path.join(dir, "descendant-survived.txt");

		// Spawn a lightweight shell fixture + descendant that ignores SIGTERM and tries to write a sentinel after 1.1s
		const treeBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":"conv-tree","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"conversation_id":"conv-tree","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"partial progress here"}}'
(trap '' TERM; sleep 1.1 && echo alive > ${JSON.stringify(sentinelPath)}) &
sleep 30
`,
			"fake-agy-tree",
		);

		const t0 = Date.now();
		const aborted = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "long job with SIGTERM-ignoring descendant",
			index: 0,
			id: "TimeoutTreeJob",
			maxRuntimeMs: 450,
			settings: makeSettings(treeBin, { "*": "gemini-4-neon-low" }),
		});
		const elapsed = Date.now() - t0;
		expect(aborted.exitCode).toBe(1);
		expect(aborted.aborted).toBe(true);
		expect(aborted.abortReason).toContain("maxRuntimeMs");
		expect(aborted.output).toContain("partial progress here");
		// Hard timeout uses immediate SIGKILL (no 1000ms SIGTERM grace period, which would exceed 1450ms)
		expect(elapsed).toBeLessThan(1_250);

		// Wait past the descendant's sleep window and verify it was hard-killed at the deadline
		await Bun.sleep(950);
		expect(await Bun.file(sentinelPath).exists()).toBe(false);

		// Successful agy exit with a lingering helper holding inherited pipes settles promptly even with unlimited runtime
		const lingerSentinel = path.join(dir, "linger-survived.txt");
		const lingerBin = await createFakeAgyShell(
			dir,
			`(trap '' TERM; sleep 3 && echo lingered > ${JSON.stringify(lingerSentinel)}) &
printf '%s\\n' '{"event":"init","conversation_id":"conv-linger","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-linger","status":"SUCCESS","response":"finished fast","num_turns":1}}'
exit 0
`,
			"fake-agy-linger",
		);
		const tLinger = Date.now();
		const settled = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "job with lingering helper",
			index: 0,
			id: "LingerHelperJob",
			maxRuntimeMs: 0,
			settings: makeSettings(lingerBin, { "*": "gemini-4-neon-low" }),
		});
		const lingerElapsed = Date.now() - tLinger;
		expect(settled.exitCode).toBe(0);
		expect(settled.aborted).toBeUndefined();
		expect(settled.output).toBe("finished fast");
		expect(lingerElapsed).toBeLessThan(1_800);
		expect(await Bun.file(lingerSentinel).exists()).toBe(false);

		const errBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-err","status":"ERROR","response":"","error":"boom from agy","num_turns":0}}'
exit 1
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
	}, 30_000);

	it("preserves nonzero exit with lingering helpers, handles missing-result helpers, reconciles explicit zeros and multi-turn context, spills oversized schemas, and restores tool start timestamps", async () => {
		const dir = await makeTempDir("agy-runner-r3-");

		// 1. SUCCESS frame followed by exit 23 WITH a lingering helper must preserve exitCode 23 (never masked to 0)
		const exit23Bin = await createFakeAgyShell(
			dir,
			`(trap '' TERM; sleep 3) &
printf '%s\\n' '{"event":"init","conversation_id":"conv-23","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-23","status":"SUCCESS","response":"said success then crashed","num_turns":1}}'
exit 23
`,
			"fake-agy-exit23",
		);
		const t23 = Date.now();
		const res23 = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "exit 23 with helper",
			index: 0,
			id: "Exit23Helper",
			maxRuntimeMs: 0,
			settings: makeSettings(exit23Bin, { "*": "gemini-4-neon-low" }),
		});
		expect(res23.exitCode).toBe(23);
		expect(res23.aborted).toBeUndefined();
		expect(Date.now() - t23).toBeLessThan(1_800);

		// 2. Exit 0 WITHOUT a terminal result frame WITH a lingering helper settles promptly under unlimited runtime
		const noResHelperBin = await createFakeAgyShell(
			dir,
			`(trap '' TERM; sleep 3) &
printf '%s\\n' '{"event":"init","conversation_id":"conv-nores","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"conversation_id":"conv-nores","step_index":1,"state":"ACTIVE","step_type":"agent_response","text_delta":"partial before drop"}}'
exit 0
`,
			"fake-agy-nores-helper",
		);
		const tNoRes = Date.now();
		const resNoRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "missing result with helper",
			index: 0,
			id: "NoResHelper",
			maxRuntimeMs: 0,
			settings: makeSettings(noResHelperBin, { "*": "gemini-4-neon-low" }),
		});
		expect(resNoRes.exitCode).toBe(1);
		expect(resNoRes.output).toBe("partial before drop");
		expect(Date.now() - tNoRes).toBeLessThan(1_800);

		// 3. Explicit terminal zeros override streamed counters; multi-turn result-only invocation leaves contextTokens unset
		const usageEdgeBin = await createFakeAgyBinary(
			dir,
			`
const printArg = process.argv.slice(2).find(a => a.startsWith("--print=")) ?? "";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-u", init: { model: "gemini-4-neon-low" } });
if (printArg.includes("ZERO_CASE")) {
	emit({ event: "step_update", step_update: { conversation_id: "conv-u", step_index: 1, state: "DONE", step_type: "agent_response", text_delta: "ok", usage: { input_tokens: 100, output_tokens: 10, cache_read_tokens: 40 } } });
	emit({ event: "result", result: { conversation_id: "conv-u", status: "SUCCESS", response: "ok", num_turns: 0, usage: { input_tokens: 0, output_tokens: 0, cache_read_tokens: 0 } } });
} else {
	emit({ event: "result", result: { conversation_id: "conv-u", status: "SUCCESS", response: "three turns", num_turns: 3, usage: { input_tokens: 1200, output_tokens: 80, cache_read_tokens: 300 } } });
}
`,
			"fake-agy-usage-edge",
		);
		const zeroRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "ZERO_CASE",
			index: 0,
			id: "ZeroTotals",
			settings: makeSettings(usageEdgeBin, { "*": "gemini-4-neon-low" }),
		});
		expect(zeroRes.exitCode).toBe(0);
		expect(zeroRes.requests).toBe(0);
		expect(zeroRes.tokens).toBe(0);
		expect(zeroRes.usage?.cacheRead).toBe(0);

		const multiTurnRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "THREE_TURN_CASE",
			index: 0,
			id: "ThreeTurnNoContext",
			settings: makeSettings(usageEdgeBin, { "*": "gemini-4-neon-low" }),
		});
		expect(multiTurnRes.exitCode).toBe(0);
		expect(multiTurnRes.requests).toBe(3);
		expect(multiTurnRes.tokens).toBe(1280);
		expect(multiTurnRes.usage?.cacheRead).toBe(300);
		expect(multiTurnRes.contextTokens).toBeUndefined();

		// 4. Short prompt + valid 75 KB root-object JSON schema spills schema into instructions instead of hitting E2BIG
		const spillSchemaBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
const argv = process.argv.slice(2);
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
const printArg = argv.find(a => a.startsWith("--print=")) ?? "";
const match = printArg.match(/from (\\S+) and execute/);
const spillContent = match ? fs.readFileSync(match[1], "utf8") : "";
const hasSchemaInSpill = spillContent.includes("<output_format>") && spillContent.includes('"summary"');
const hasFlag = argv.includes("--json-schema");
emit({ event: "init", conversation_id: "conv-bigschema", init: { model: "gemini-4-neon-low" } });
if (!hasFlag && hasSchemaInSpill) {
	emit({ event: "result", result: { conversation_id: "conv-bigschema", status: "SUCCESS", response: '{"summary":"validated"}', num_turns: 1 } });
} else {
	emit({ event: "result", result: { conversation_id: "conv-bigschema", status: "SUCCESS", response: "plain prose", num_turns: 1 } });
}
`,
			"fake-agy-bigschema",
		);
		const hugeObjectSchema = {
			type: "object",
			description: "d\n".repeat(38_000),
			properties: { summary: { type: "string" } },
			required: ["summary"],
		};
		const bigSchemaRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "short prompt",
			index: 0,
			id: "BigSchema",
			artifactsDir: dir,
			outputSchema: hugeObjectSchema,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			settings: makeSettings(spillSchemaBin, { "*": "gemini-4-neon-low" }),
		});
		expect(bigSchemaRes.exitCode).toBe(0);
		expect(bigSchemaRes.structuredOutput).toEqual({
			source: "caller",
			mode: "strict",
			status: "valid",
			data: { summary: "validated" },
		});

		// 5. Completing tool B while tool A is still active restores tool A's original start timestamp
		const timerBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-timer", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-timer", step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "tool_a", tool_info: { name: "tool_a", parameters: { Path: "/tmp/a" } } } });
await Bun.sleep(180);
emit({ event: "step_update", step_update: { conversation_id: "conv-timer", step_index: 2, state: "ACTIVE", step_type: "tool", tool_name: "tool_b", tool_info: { name: "tool_b", parameters: { Path: "/tmp/b" } } } });
await Bun.sleep(180);
emit({ event: "step_update", step_update: { conversation_id: "conv-timer", step_index: 2, state: "DONE", step_type: "tool", tool_name: "tool_b", tool_info: { name: "tool_b", output: "b done" } } });
await Bun.sleep(180);
emit({ event: "step_update", step_update: { conversation_id: "conv-timer", step_index: 1, state: "DONE", step_type: "tool", tool_name: "tool_a", tool_info: { name: "tool_a", output: "a done" } } });
emit({ event: "result", result: { conversation_id: "conv-timer", status: "SUCCESS", response: "done", num_turns: 1 } });
`,
			"fake-agy-timer",
		);
		const snapshots: Array<{ tool?: string; startMs?: number }> = [];
		const timerRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "interleaved timers",
			index: 0,
			id: "InterleavedTimers",
			settings: makeSettings(timerBin, { "*": "gemini-4-neon-low" }),
			onProgress: p => snapshots.push({ tool: p.currentTool, startMs: p.currentToolStartMs }),
		});
		expect(timerRes.exitCode).toBe(0);
		const firstAIdx = snapshots.findIndex(s => s.tool === "tool_a");
		const firstBIdx = snapshots.findIndex((s, idx) => idx > firstAIdx && s.tool === "tool_b");
		const restoredAIdx = snapshots.findIndex((s, idx) => idx > firstBIdx && s.tool === "tool_a");
		expect(firstAIdx).toBeGreaterThanOrEqual(0);
		expect(firstBIdx).toBeGreaterThan(firstAIdx);
		expect(restoredAIdx).toBeGreaterThan(firstBIdx);
		const firstA = snapshots[firstAIdx]!;
		const firstB = snapshots[firstBIdx]!;
		const restoredA = snapshots[restoredAIdx]!;
		expect(firstA.startMs).toBeDefined();
		expect(firstB.startMs).toBeDefined();
		expect(firstB.startMs!).toBeGreaterThan(firstA.startMs!);
		expect(restoredA.startMs).toBe(firstA.startMs);

		// 6. A throwing onToolResult callback must NOT replay the frame, suppress live tool-completion progress, or leak an unhandled rejection
		const throwingBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-throw", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-throw", step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", parameters: { Path: "/tmp/x" } } } });
await Bun.sleep(180);
emit({ event: "step_update", step_update: { conversation_id: "conv-throw", step_index: 1, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { name: "read_file", output: "tool payload" } } });
await Bun.sleep(250);
emit({ event: "result", result: { conversation_id: "conv-throw", status: "SUCCESS", response: "completed after throw", num_turns: 1 } });
`,
			"fake-agy-throw-cb",
		);
		const unhandled: unknown[] = [];
		const onUnhandled = (reason: unknown) => {
			unhandled.push(reason);
		};
		process.on("unhandledRejection", onUnhandled);
		let cbCalls = 0;
		const cbProgress: Array<{ status: string; currentTool?: string; toolCount: number }> = [];
		try {
			const cbRes = await runSubprocess({
				cwd: dir,
				agent: makeAgent(),
				task: "callback throws",
				index: 0,
				id: "ThrowCallbackJob",
				settings: makeSettings(throwingBin, { "*": "gemini-4-neon-low" }),
				onProgress: p => cbProgress.push({ status: p.status, currentTool: p.currentTool, toolCount: p.toolCount }),
				onToolResult: () => {
					cbCalls += 1;
					// Null-prototype object throws on String(err); live scheduleProgress() must still run in finally
					throw Object.create(null);
				},
			});
			await Bun.sleep(20);
			expect(cbRes.exitCode).toBe(0);
			expect(cbRes.output).toBe("completed after throw");
			expect(cbCalls).toBe(1);
			expect(unhandled).toEqual([]);
			// Live running frame clearing currentTool must still be published despite throwing onToolResult
			expect(cbProgress.some(p => p.status === "running" && p.currentTool === undefined && p.toolCount === 1)).toBe(
				true,
			);
		} finally {
			process.off("unhandledRejection", onUnhandled);
		}
	}, 30_000);
});
