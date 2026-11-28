import { afterEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { Settings } from "../../src/config/settings";
import { IrcBus } from "../../src/irc/bus";
import { handleRpcCancelSubagent, handleRpcSteerSubagent } from "../../src/modes/rpc/rpc-mode";
import { RpcSubagentRegistry } from "../../src/modes/rpc/rpc-subagents";
import { AgentLifecycleManager } from "../../src/registry/agent-lifecycle";
import { AgentRegistry, MAIN_AGENT_ID } from "../../src/registry/agent-registry";
import { isLocalAgentSession } from "../../src/session/agent-session-handle";
import { CostGate } from "../../src/session/cost-gate";
import { resolveAgySubagentConfig } from "../../src/task/agy-cli-runner";
import {
	type ExecutorOptions,
	runSubagentFollowUpTurn,
	runSubprocess,
	type SubagentToolResult,
} from "../../src/task/executor";
import { runIsolatedSubprocess } from "../../src/task/isolation-runner";
import type { AgentProgress } from "../../src/task/task-details";
import type { AgentDefinition, SubagentLifecyclePayload, SubagentProgressPayload } from "../../src/task/types";
import { TASK_SUBAGENT_LIFECYCLE_CHANNEL, TASK_SUBAGENT_PROGRESS_CHANNEL } from "../../src/task/types";
import type { ThreadService } from "../../src/thread/services";
import { agentsThreadService } from "../../src/thread/services/agents";
import { identityThreadService } from "../../src/thread/services/identity";
import { SubagentWorker } from "../../src/thread/subagent-worker";
import { EventBus } from "../../src/utils/event-bus";
import type { AgentsProbe, AgentsProbeGlobals } from "../fixtures/agents-bridge-worker";

const tempDirs: string[] = [];

/** Helper snippet embedded in fake Bun `agy` binaries to read the first NDJSON user prompt from stdin. */
const READ_FIRST_STDIN_PROMPT_SNIPPET = `
async function readFirstPrompt() {
	const reader = Bun.stdin.stream().getReader();
	const decoder = new TextDecoder();
	let buf = "";
	while (true) {
		const { value, done } = await reader.read();
		if (value) buf += decoder.decode(value, { stream: true });
		const nl = buf.indexOf("\\n");
		if (nl !== -1) {
			const line = buf.slice(0, nl).trim();
			reader.releaseLock();
			if (!line) return "";
			try {
				const obj = JSON.parse(line);
				return typeof obj?.message?.content === "string" ? obj.message.content : "";
			} catch {
				return line;
			}
		}
		if (done) {
			reader.releaseLock();
			const trimmed = (buf + decoder.decode()).trim();
			if (!trimmed) return "";
			try {
				const obj = JSON.parse(trimmed);
				return typeof obj?.message?.content === "string" ? obj.message.content : "";
			} catch {
				return trimmed;
			}
		}
	}
}
`;

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
	await AgentLifecycleManager.global().dispose();
	AgentLifecycleManager.resetGlobalForTests();
	AgentRegistry.resetGlobalForTests();
	IrcBus.resetGlobalForTests();
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

		// keepAlive subagents now route to agy (with live registry + mid-job messaging + revival);
		// seeded topic lifecycles and OMP-only contracts still stay on native OMP.
		expect(resolveAgySubagentConfig(base({ keepAlive: true }))).toEqual({
			binary: bin,
			model: "gemini-4-neon-medium",
		});
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
		const stdinDump = path.join(dir, "stdin.txt");
		// Seed a stale JSON sidecar from an earlier run to prove plain replacement clears it
		const staleSidecar = path.join(artifactsDir, "WorkerDemo.json");
		await fs.writeFile(staleSidecar, '{"stale":true}\n');

		const bin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
fs.writeFileSync(${JSON.stringify(argvDump)}, JSON.stringify(process.argv.slice(2)));
const firstPrompt = await readFirstPrompt();
fs.writeFileSync(${JSON.stringify(stdinDump)}, firstPrompt);
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
		expect(argv).toContain("--input-format");
		expect(argv).toContain("--output-format");
		expect(argv).toContain("stream-json");
		expect(argv).toContain("--dangerously-skip-permissions");
		expect(argv).toContain("--model");
		expect(argv).toContain("gemini-4-neon-medium");
		expect(argv.some(a => a.startsWith("--print="))).toBe(false);
		const sentPrompt = await fs.readFile(stdinDump, "utf8");
		expect(sentPrompt).toContain("Shared repo context");
		expect(sentPrompt).toContain("--check-leading-dash find target");
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
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const idx = argv.indexOf("--json-schema");
const promptText = await readFirstPrompt();
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
} else if (promptText.includes("<output_format>") && promptText.includes("raw valid JSON")) {
	const payload = promptText.includes('"type":"array"') ? '["a","b"]' : '{"summary":"all good"}';
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

	it("accumulates multi-step usage per step, streams huge prompts over stdin without E2BIG, and settles cleanly on missing result", async () => {
		const dir = await makeTempDir("agy-runner-steps-");
		const logPath = path.join(dir, "invocations.jsonl");

		// Step 1 reports partial step_update usage (1st DONE has usage, 2nd DONE omits usage) + aggregate on result;
		// Step 2 reports usage ONLY on result frame.
		const bin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
fs.appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ conv, argvLen: JSON.stringify(argv).length, promptLen: promptText.length }) + "\\n");
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

		// Huge prompts (> 100 KiB) travel cleanly over NDJSON stdin without touching argv or temp spill files
		const hugePrompt = "a\n".repeat(55_000);
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
		// Authoritative terminal totals per step: 2+2=4 requests, 330+220=550 tokens, 90+600=690 cacheRead
		expect(result.requests).toBe(4);
		expect(result.tokens).toBe(550);
		expect(result.usage?.cacheRead).toBe(690);

		const calls = (await fs.readFile(logPath, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as { conv: string | null; argvLen: number; promptLen: number });
		expect(calls.length).toBe(2);
		expect(calls[0]?.conv).toBeNull();
		expect(calls[0]?.argvLen).toBeLessThan(1024);
		expect(calls[0]?.promptLen).toBeGreaterThan(100_000);
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
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const promptText = await readFirstPrompt();
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-u", init: { model: "gemini-4-neon-low" } });
if (promptText.includes("ZERO_CASE")) {
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

		// 4. Short prompt + valid 75 KB root-object JSON schema folds schema into stdin prompt instead of hitting E2BIG
		const spillSchemaBin = await createFakeAgyBinary(
			dir,
			`
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
const promptText = await readFirstPrompt();
const hasSchemaInPrompt = promptText.includes("<output_format>") && promptText.includes('"summary"');
const hasFlag = argv.includes("--json-schema");
emit({ event: "init", conversation_id: "conv-bigschema", init: { model: "gemini-4-neon-low" } });
if (!hasFlag && hasSchemaInPrompt) {
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

		// 7. Isolated agy task with omitted keepAlive must NOT tear down the workspace before run-end patch capture
		const isoRoot = await makeTempDir("agy-runner-iso-repo-");
		const isoArtifacts = path.join(isoRoot, "artifacts");
		await fs.mkdir(isoArtifacts, { recursive: true });
		const initGit = Bun.spawnSync(["git", "init", "-q"], { cwd: isoRoot });
		expect(initGit.exitCode).toBe(0);
		await fs.writeFile(path.join(isoRoot, "base.txt"), "initial\n");
		Bun.spawnSync(["git", "config", "user.email", "test@example.com"], { cwd: isoRoot });
		Bun.spawnSync(["git", "config", "user.name", "Test"], { cwd: isoRoot });
		Bun.spawnSync(["git", "add", "base.txt"], { cwd: isoRoot });
		Bun.spawnSync(["git", "commit", "-q", "-m", "init"], { cwd: isoRoot });

		const isoWorkspaceDir = path.join(isoRoot, "isolated-wt");
		const isoWorktreeAdd = Bun.spawnSync(["git", "worktree", "add", "-q", "--detach", isoWorkspaceDir], {
			cwd: isoRoot,
		});
		expect(isoWorktreeAdd.exitCode).toBe(0);
		const isoWriterBin = await createFakeAgyShell(
			dir,
			`printf 'from-agy\\n' > created.txt
printf '%s\\n' '{"event":"init","conversation_id":"conv-iso","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-iso","status":"SUCCESS","response":"wrote created.txt","num_turns":1}}'
exit 0
`,
			"fake-agy-iso-writer",
		);
		const order: string[] = [];
		const isoResult = await runIsolatedSubprocess({
			context: { repoRoot: isoRoot },
			agentId: "IsoAgyWorker",
			artifactsDir: isoArtifacts,
			mergeMode: "patch",
			openWorkspace: async () => ({
				dir: isoWorkspaceDir,
				backend: "rcopy",
				cleanup: async () => {
					order.push("cleanup");
					await fs.rm(isoWorkspaceDir, { recursive: true, force: true });
				},
				retain: async () => ({ dir: isoWorkspaceDir, sidecarOk: true }),
			}),
			baseOptions: {
				cwd: isoRoot,
				agent: makeAgent(),
				task: "write created.txt",
				index: 0,
				id: "IsoAgyWorker",
				artifactsDir: isoArtifacts,
				settings: makeSettings(isoWriterBin, { "*": "gemini-4-neon-low" }),
			},
			buildFailureResult: err => ({
				index: 0,
				id: "IsoAgyWorker",
				agent: "task",
				agentSource: "bundled",
				task: "write created.txt",
				exitCode: 1,
				output: "",
				stderr: String(err),
				truncated: false,
				durationMs: 0,
				tokens: 0,
				requests: 0,
				error: String(err),
			}),
		});
		expect(isoResult.exitCode).toBe(0);
		expect(isoResult.error).toBeUndefined();
		expect(isoResult.patchPath).toBeDefined();
		const capturedPatch = await fs.readFile(isoResult.patchPath!, "utf8");
		expect(capturedPatch).toContain("created.txt");
		expect(capturedPatch).toContain("+from-agy");
		expect(order).toEqual(["cleanup"]);
	}, 30_000);

	it("maps malformed agy terminal frames to failed SingleResult and lifecycle events instead of rejecting", async () => {
		const dir = await makeTempDir("omp-agy-malformed-");
		const cases = [
			{
				name: "null-response",
				frame: '{"event":"result","result":{"conversation_id":"conv-bad","status":"SUCCESS","response":null}}',
				expectedError: /non-string response/i,
			},
			{
				name: "null-result",
				frame: '{"event":"result","result":null}',
				expectedError: /missing result object/i,
			},
			{
				name: "non-string-status",
				frame: '{"event":"result","result":{"status":200,"response":"ok"}}',
				expectedError: /non-string status/i,
			},
			{
				name: "non-string-error",
				frame: '{"event":"result","result":{"status":"ERROR","error":123}}',
				expectedError: /non-string error/i,
			},
		] as const;

		for (const tc of cases) {
			const bin = await createFakeAgyShell(
				dir,
				`printf '%s\\n' '{"event":"init","conversation_id":"conv-bad","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '${tc.frame}'
exit 0
`,
				`fake-agy-${tc.name}`,
			);
			const eventBus = new EventBus();
			const lifecycle: string[] = [];
			eventBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, payload => {
				lifecycle.push((payload as SubagentLifecyclePayload).status);
			});
			const result = await runSubprocess({
				cwd: dir,
				agent: makeAgent(),
				task: "check malformed frame",
				index: 0,
				id: `Malformed-${tc.name}`,
				eventBus,
				settings: makeSettings(bin, { "*": "gemini-4-neon-low" }),
			});
			expect(result.exitCode).toBe(1);
			expect(typeof result.output).toBe("string");
			expect(result.error).toMatch(tc.expectedError);
			expect(lifecycle).toEqual(["started", "failed"]);
		}

		// Control: malformed non-terminal step_update fields are safely ignored and a valid terminal frame still succeeds.
		const resilientBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":123,"init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":0,"state":"DONE","tool_name":"read","tool_info":{"parameters":"bad","output":{"nested":true}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"DONE","text_delta":42,"usage":"bad"}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-ok","status":"SUCCESS","response":"valid-after-noise","num_turns":1}}'
exit 0
`,
			"fake-agy-resilient",
		);
		const okBus = new EventBus();
		const okLifecycle: string[] = [];
		okBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, payload => {
			okLifecycle.push((payload as SubagentLifecyclePayload).status);
		});
		const okResult = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "check resilient stream",
			index: 0,
			id: "Malformed-resilient",
			eventBus: okBus,
			settings: makeSettings(resilientBin, { "*": "gemini-4-neon-low" }),
		});
		expect(okResult.exitCode).toBe(0);
		expect(okResult.output).toBe("valid-after-noise");
		expect(okResult.error).toBeUndefined();
		expect(okLifecycle).toEqual(["started", "completed"]);

		// Observer isolation: throwing onProgress (including Error with throwing message getter),
		// async rejecting onProgress, and a throwing primary session bus emit must not suppress
		// observability bus publication, produce unhandled rejections, or replace the settled SingleResult.
		const unhandled: unknown[] = [];
		const onUnhandled = (reason: unknown) => {
			unhandled.push(reason);
		};
		process.on("unhandledRejection", onUnhandled);
		try {
			for (const scenario of [
				{ bin: resilientBin, expectedExit: 0, expectedLifecycle: "completed" as const },
				{ bin: path.join(dir, "fake-agy-null-response"), expectedExit: 1, expectedLifecycle: "failed" as const },
			]) {
				const sessionBus = new EventBus();
				const origEmit = sessionBus.emit;
				sessionBus.emit = (channel, data) => {
					if (
						channel === TASK_SUBAGENT_LIFECYCLE_CHANNEL &&
						(data as SubagentLifecyclePayload).status !== "started"
					) {
						throw new Error("primary bus failed on terminal emit");
					}
					origEmit(channel, data);
				};
				const obsBus = new EventBus();
				const obsLifecycle: string[] = [];
				const obsProgressStatuses: string[] = [];
				obsBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, payload => {
					obsLifecycle.push((payload as SubagentLifecyclePayload).status);
				});
				obsBus.on(TASK_SUBAGENT_PROGRESS_CHANNEL, payload => {
					obsProgressStatuses.push((payload as SubagentProgressPayload).progress.status);
				});
				const hostileError = new Error("hidden");
				Object.defineProperty(hostileError, "message", {
					get() {
						throw new Error("message getter exploded");
					},
				});
				const res = await runSubprocess({
					cwd: dir,
					agent: makeAgent(),
					task: "check hostile progress observer and failing primary bus",
					index: 0,
					id: `HostileObserver-${scenario.expectedLifecycle}`,
					eventBus: sessionBus,
					subagentEventBus: obsBus,
					onProgress: () => {
						throw hostileError;
					},
					settings: makeSettings(scenario.bin, { "*": "gemini-4-neon-low" }),
				});
				expect(res.exitCode).toBe(scenario.expectedExit);
				expect(obsLifecycle).toEqual(["started", scenario.expectedLifecycle]);
				expect(obsProgressStatuses).toContain("running");
				expect(obsProgressStatuses.at(-1)).toBe(scenario.expectedLifecycle);
			}

			// Async rejecting onProgress (native async, virtual-then Proxy, and Promise with shadowed catch)
			// must not trigger unhandledRejection or fail the settled run.
			let proxyThenCalls = 0;
			const asyncCases: Array<{ name: string; observer: (progress: AgentProgress) => unknown }> = [
				{
					name: "native-async",
					observer: async p => {
						throw new Error(`async observer failed on ${p.status}`);
					},
				},
				{
					name: "virtual-then-proxy",
					observer: p =>
						new Proxy(
							{},
							{
								has: () => false,
								get: (_target, prop) =>
									prop === "then"
										? (_resolve: unknown, reject: (reason: unknown) => void) => {
												proxyThenCalls += 1;
												queueMicrotask(() => reject(new Error(`proxy thenable failed on ${p.status}`)));
											}
										: undefined,
							},
						),
				},
				{
					name: "shadowed-catch-promise",
					observer: p => {
						const rejected = Promise.reject(new Error(`shadowed-catch failed on ${p.status}`));
						Object.defineProperty(rejected, "catch", { value: undefined });
						return rejected;
					},
				},
			];
			for (const ac of asyncCases) {
				const asyncBus = new EventBus();
				const asyncLifecycle: string[] = [];
				asyncBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, payload => {
					asyncLifecycle.push((payload as SubagentLifecyclePayload).status);
				});
				const asyncRes = await runSubprocess({
					cwd: dir,
					agent: makeAgent(),
					task: `check ${ac.name} progress observer`,
					index: 0,
					id: `AsyncObserver-${ac.name}`,
					eventBus: asyncBus,
					onProgress: ac.observer as unknown as (progress: AgentProgress) => void,
					settings: makeSettings(resilientBin, { "*": "gemini-4-neon-low" }),
				});
				await Bun.sleep(20);
				expect(asyncRes.exitCode).toBe(0);
				expect(asyncLifecycle).toEqual(["started", "completed"]);
			}
			expect(proxyThenCalls).toBeGreaterThanOrEqual(2);

			// Both synchronously throwing and asynchronously rejecting onRelease hooks must preserve
			// the settled SingleResult and terminal lifecycle publication.
			for (const releaseMode of ["sync-throw", "async-reject"] as const) {
				const releaseBus = new EventBus();
				const releaseLifecycle: string[] = [];
				releaseBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, payload => {
					releaseLifecycle.push((payload as SubagentLifecyclePayload).status);
				});
				let releaseCalled = 0;
				const releaseRes = await runSubprocess({
					cwd: dir,
					agent: makeAgent(),
					task: `check ${releaseMode} release hook`,
					index: 0,
					id: `ReleaseHook-${releaseMode}`,
					keepAlive: false,
					eventBus: releaseBus,
					onRelease:
						releaseMode === "sync-throw"
							? () => {
									releaseCalled += 1;
									throw new Error("sync release hook failed");
								}
							: async () => {
									releaseCalled += 1;
									throw new Error("async release hook failed");
								},
					settings: makeSettings(resilientBin, { "*": "gemini-4-neon-low" }),
				});
				expect(releaseCalled).toBe(1);
				expect(releaseRes.exitCode).toBe(0);
				expect(releaseRes.output).toBe("valid-after-noise");
				expect(releaseLifecycle).toEqual(["started", "completed"]);
			}
			expect(unhandled).toEqual([]);
		} finally {
			process.off("unhandledRejection", onUnhandled);
		}
	}, 30_000);

	it("streams Herdr-compatible child JSONL transcripts with model_change, tool calls, and session_exit", async () => {
		const dir = await makeTempDir("omp-agy-transcript-");
		const parentSession = path.join(dir, "parent-session.jsonl");
		const artifactsDir = path.join(dir, "parent-session");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p1" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const toolBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":"conv-tr","init":{"model":"gemini-4-neon-medium"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"ACTIVE","text_delta":"Checking files..."}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":1,"state":"ACTIVE","tool_name":"read_file","tool_info":{"parameters":{"Path":"src/app.ts"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":1,"state":"DONE","tool_name":"read_file","tool_info":{"parameters":{"Path":"src/app.ts"},"output":"export const x = 1;"}}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-tr","status":"SUCCESS","response":"All checked.","num_turns":2}}'
exit 0
`,
			"fake-agy-transcript",
		);

		const bus = new EventBus();
		const lifecycles: SubagentLifecyclePayload[] = [];
		bus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, p => lifecycles.push(p as SubagentLifecyclePayload));

		const okRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Inspect src/app.ts and report",
			index: 0,
			id: "TranscriptOk",
			artifactsDir,
			sessionFile: parentSession,
			eventBus: bus,
			settings: makeSettings(toolBin, { task: "default" }),
		});
		expect(okRes.exitCode).toBe(0);

		const transcriptPath = path.join(artifactsDir, "TranscriptOk.jsonl");
		expect(lifecycles.map(l => [l.status, l.sessionFile])).toEqual([
			["started", transcriptPath],
			["completed", transcriptPath],
		]);

		const lines = (await fs.readFile(transcriptPath, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		// Header must link back to exact parentSession for Herdr child_session() validation.
		expect(lines[0]?.type).toBe("session");
		expect(lines[0]?.version).toBe(3);
		expect(lines[0]?.parentSession).toBe(parentSession);

		// All non-header entries must form an intact linked parentId chain with unique ids.
		let prevId: string | null = null;
		for (const entry of lines.slice(1)) {
			expect(typeof entry.id).toBe("string");
			expect((entry.id as string).length).toBeGreaterThan(0);
			expect(entry.parentId).toBe(prevId);
			prevId = entry.id as string;
		}

		// Model changes include initial agy-cli/default and refined agy-cli/gemini-4-neon-medium.
		const modelChanges = lines.filter(l => l.type === "model_change").map(l => l.model);
		expect(modelChanges).toEqual(["agy-cli/default", "agy-cli/gemini-4-neon-medium"]);

		// Contains user prompt, assistant toolCall + tool_execution_start intent, toolResult, final stop, and session_exit.
		expect(
			lines.some(
				l =>
					l.type === "message" &&
					(l.message as Record<string, unknown>).role === "user" &&
					JSON.stringify(l.message).includes("Inspect src/app.ts and report"),
			),
		).toBe(true);
		const toolStart = lines.find(l => l.type === "custom" && l.customType === "tool_execution_start");
		expect(toolStart).toBeDefined();
		expect((toolStart?.data as Record<string, unknown>)?.intent).toBe("read_file: src/app.ts");
		expect(
			lines.some(
				l =>
					l.type === "message" &&
					(l.message as Record<string, unknown>).role === "toolResult" &&
					(l.message as Record<string, unknown>).toolName === "read_file" &&
					JSON.stringify(l.message).includes("export const x = 1;"),
			),
		).toBe(true);

		const lastMessage = lines.at(-2);
		const lastEntry = lines.at(-1);
		expect(lastMessage?.type).toBe("message");
		expect((lastMessage?.message as Record<string, unknown>)?.stopReason).toBe("stop");
		expect(lastEntry?.type).toBe("custom");
		expect(lastEntry?.customType).toBe("session_exit");

		// Failure path writes stopReason="error" before session_exit so Herdr marks it failed.
		const failBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":"conv-fail","init":{"model":"gemini-4-neon-low"}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-fail","status":"ERROR","error":"boom"}}'
exit 1
`,
			"fake-agy-transcript-fail",
		);
		const failRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "fail cleanly",
			index: 0,
			id: "TranscriptFail",
			artifactsDir,
			sessionFile: parentSession,
			settings: makeSettings(failBin, { task: "gemini-4-neon-low" }),
		});
		expect(failRes.exitCode).toBe(1);
		const failLines = (await fs.readFile(path.join(artifactsDir, "TranscriptFail.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		expect((failLines.at(-2)?.message as Record<string, unknown>)?.stopReason).toBe("error");
		expect(failLines.at(-1)?.customType).toBe("session_exit");

		// Path traversal in id is safely refused without failing the subagent run.
		const traversalRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "traversal guard",
			index: 0,
			id: "../escape",
			artifactsDir,
			sessionFile: parentSession,
			settings: makeSettings(toolBin, { task: "gemini-4-neon-medium" }),
		});
		expect(traversalRes.exitCode).toBe(0);
		expect(await Bun.file(path.join(dir, "escape.jsonl")).exists()).toBe(false);

		// Unwritable artifactsDir (file where directory expected) disables transcript cleanly without failing the run.
		const blockingFile = path.join(dir, "not-a-dir");
		await fs.writeFile(blockingFile, "blocker");
		const unwritableBus = new EventBus();
		const unwritableLifecycles: SubagentLifecyclePayload[] = [];
		unwritableBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, p => unwritableLifecycles.push(p as SubagentLifecyclePayload));
		const unwritableRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "unwritable artifactsDir",
			index: 0,
			id: "TranscriptUnwritable",
			artifactsDir: blockingFile,
			sessionFile: parentSession,
			eventBus: unwritableBus,
			settings: makeSettings(toolBin, { task: "gemini-4-neon-medium" }),
		});
		expect(unwritableRes.exitCode).toBe(0);
		expect(unwritableLifecycles.map(l => [l.status, l.sessionFile])).toEqual([
			["started", undefined],
			["completed", undefined],
		]);
	}, 30_000);

	it("fixes tool correlation, exclusive transcript creation, streamed fallback dedup, and per-turn usage recording", async () => {
		const dir = await makeTempDir("omp-agy-p2-usage-");
		const parentSession = path.join(dir, "parent-session.jsonl");
		const artifactsDir = path.join(dir, "parent-session");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p1" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		// 1. Out-of-order unindexed tool completions match by tool name, repeated indexed ACTIVE is idempotent,
		// and per-turn + residual usage is recorded without double-counting.
		const multiBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":"conv-p2","init":{"model":"gemini-4-neon-medium"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"ACTIVE","text_delta":"Inspecting repo. "}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"DONE","usage":{"input_tokens":100,"output_tokens":20,"cache_read_tokens":50},"duration_seconds":0.25}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":7,"state":"ACTIVE","tool_name":"bash","tool_info":{"parameters":{"command":"ls"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":7,"state":"ACTIVE","tool_name":"bash","tool_info":{"parameters":{"command":"ls -la"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":7,"state":"DONE","tool_name":"bash","tool_info":{"parameters":{"command":"ls -la"},"output":"file.ts"}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","state":"ACTIVE","tool_name":"read","tool_info":{"parameters":{"path":"a.ts"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","state":"ACTIVE","tool_name":"grep","tool_info":{"parameters":{"pattern":"foo"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","state":"DONE","tool_name":"grep","tool_info":{"output":"grep-out"}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","state":"DONE","tool_name":"read","tool_info":{"output":"read-out"}}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-p2","status":"SUCCESS","response":"Final summary.","duration_seconds":1.2,"num_turns":2,"usage":{"input_tokens":350,"output_tokens":70,"cache_read_tokens":200}}}'
exit 0
`,
			"fake-agy-p2-multi",
		);

		const res = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Run tools and summarize",
			index: 0,
			id: "ToolOrderAndUsage",
			artifactsDir,
			sessionFile: parentSession,
			settings: makeSettings(multiBin, { task: "default" }),
		});
		expect(res.exitCode).toBe(0);

		const transcriptPath = path.join(artifactsDir, "ToolOrderAndUsage.jsonl");
		const entries = (await fs.readFile(transcriptPath, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);

		// Delegated user prompt carries attribution="agent".
		const userEntry = entries.find(
			e => e.type === "message" && (e.message as Record<string, unknown>).role === "user",
		);
		expect((userEntry?.message as Record<string, unknown>)?.attribution).toBe("agent");

		// agy_session custom entry captures conversationId once.
		const agySessions = entries.filter(e => e.type === "custom" && e.customType === "agy_session");
		expect(agySessions).toHaveLength(1);
		expect((agySessions[0]?.data as Record<string, unknown>)?.conversationId).toBe("conv-p2");

		// Duplicate indexed ACTIVE for bash (step_index 7) minted only ONE tool_execution_start.
		const toolStarts = entries.filter(e => e.type === "custom" && e.customType === "tool_execution_start");
		const startByName = new Map(
			toolStarts.map(e => [
				(e.data as Record<string, unknown>).toolName as string,
				(e.data as Record<string, unknown>).toolCallId as string,
			]),
		);
		expect(toolStarts).toHaveLength(3);

		// Unindexed out-of-order DONE grep / DONE read attach each output to its matching tool call ID.
		const toolResults = entries
			.filter(e => e.type === "message" && (e.message as Record<string, unknown>).role === "toolResult")
			.map(e => e.message as Record<string, unknown>);
		const grepResult = toolResults.find(r => r.toolName === "grep");
		const readResult = toolResults.find(r => r.toolName === "read");
		expect(grepResult?.toolCallId).toBe(startByName.get("grep"));
		expect(JSON.stringify(grepResult?.content)).toContain("grep-out");
		expect(readResult?.toolCallId).toBe(startByName.get("read"));
		expect(JSON.stringify(readResult?.content)).toContain("read-out");

		// Assistant messages carry api="agy-cli", provider="agy-cli", bare model, and exact non-overlapping usage totals.
		const assistantUsageMsgs = entries
			.filter(
				e =>
					e.type === "message" &&
					(e.message as Record<string, unknown>).role === "assistant" &&
					(e.message as Record<string, unknown>).usage !== undefined,
			)
			.map(e => e.message as Record<string, unknown>);
		expect(assistantUsageMsgs).toHaveLength(2);
		for (const msg of assistantUsageMsgs) {
			expect(msg.api).toBe("agy-cli");
			expect(msg.provider).toBe("agy-cli");
			expect(msg.model).toBe("gemini-4-neon-medium");
		}
		const sumInput = assistantUsageMsgs.reduce((acc, m) => acc + ((m.usage as Record<string, number>).input ?? 0), 0);
		const sumOutput = assistantUsageMsgs.reduce(
			(acc, m) => acc + ((m.usage as Record<string, number>).output ?? 0),
			0,
		);
		const sumCacheRead = assistantUsageMsgs.reduce(
			(acc, m) => acc + ((m.usage as Record<string, number>).cacheRead ?? 0),
			0,
		);
		expect({ sumInput, sumOutput, sumCacheRead }).toEqual({ sumInput: 350, sumOutput: 70, sumCacheRead: 200 });

		// 2. Exclusive open (`wx`) refuses to overwrite an existing transcript and omits sessionFile from lifecycle.
		const existingPath = path.join(artifactsDir, "ExistingTranscript.jsonl");
		const sentinelContent = '{"type":"session","id":"do-not-overwrite"}\n';
		await fs.writeFile(existingPath, sentinelContent);
		const existingBus = new EventBus();
		const existingLifecycles: SubagentLifecyclePayload[] = [];
		existingBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, p => existingLifecycles.push(p as SubagentLifecyclePayload));
		const existingRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "must not clobber",
			index: 0,
			id: "ExistingTranscript",
			artifactsDir,
			sessionFile: parentSession,
			eventBus: existingBus,
			settings: makeSettings(multiBin, { task: "default" }),
		});
		expect(existingRes.exitCode).toBe(0);
		expect(await fs.readFile(existingPath, "utf8")).toBe(sentinelContent);
		expect(existingLifecycles.map(l => [l.status, l.sessionFile])).toEqual([
			["started", undefined],
			["completed", undefined],
		]);

		// 3. Streamed fallback when `result.response` is omitted does NOT replay pre-tool text in the final transcript turn.
		const fallbackBin = await createFakeAgyShell(
			dir,
			`printf '%s\\n' '{"event":"init","conversation_id":"conv-fb","init":{"model":"gemini-4-neon-medium"}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"ACTIVE","text_delta":"I will inspect."}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":1,"state":"ACTIVE","tool_name":"read","tool_info":{"parameters":{"path":"x.ts"}}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"tool","step_index":1,"state":"DONE","tool_name":"read","tool_info":{"output":"ok"}}}'
printf '%s\\n' '{"event":"step_update","step_update":{"step_type":"agent_response","state":"ACTIVE","text_delta":"Tail answer only."}}'
printf '%s\\n' '{"event":"result","result":{"conversation_id":"conv-fb","status":"SUCCESS","num_turns":2}}'
exit 0
`,
			"fake-agy-fallback-dedup",
		);
		const fbRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "fallback dedup check",
			index: 0,
			id: "FallbackDedup",
			artifactsDir,
			sessionFile: parentSession,
			settings: makeSettings(fallbackBin, { task: "default" }),
		});
		expect(fbRes.exitCode).toBe(0);
		// SingleResult still receives the full streamed aggregate, while the transcript records pre-tool text only once.
		expect(fbRes.output).toContain("I will inspect.");
		expect(fbRes.output).toContain("Tail answer only.");
		const fbEntries = (await fs.readFile(path.join(artifactsDir, "FallbackDedup.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		const finalAssistant = fbEntries.at(-2)?.message as Record<string, unknown>;
		expect(JSON.stringify(finalAssistant.content)).toContain("Tail answer only.");
		expect(JSON.stringify(finalAssistant.content)).not.toContain("I will inspect.");
	}, 30_000);

	it("delivers mid-job queued follow-ups over live stream-json stdin in the same agy process without interrupting turn 1", async () => {
		const dir = await makeTempDir("omp-agy-midjob-followup-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-fu" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });
		const logFile = path.join(dir, "turns.jsonl");

		// Stateful stream-json fake agy: reads NDJSON user turns until stdin EOF and runs each in one process.
		const streamBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
const reader = Bun.stdin.stream().getReader();
const decoder = new TextDecoder();
let buf = "";
let turn = 0;
const cid = "conv-live-stream";
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });

async function handleTurn(content) {
	turn += 1;
	fs.appendFileSync(${JSON.stringify(logFile)}, JSON.stringify({ pid: process.pid, turn, content }) + "\\n");
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: turn, state: "ACTIVE", step_type: "tool", tool_name: "read_file", tool_info: { parameters: { Path: "turn-" + turn } } } });
	await Bun.sleep(turn === 1 ? 260 : 80);
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: turn, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { output: "ok-" + turn } } });
	emit({
		event: "result",
		result: {
			conversation_id: cid,
			status: "SUCCESS",
			response: "reply-turn-" + turn + ":" + content.slice(-30),
			num_turns: turn,
			usage: { input_tokens: 100 * turn, output_tokens: 20 * turn, cache_read_tokens: 50 * turn },
		},
	});
}

while (true) {
	const { value, done } = await reader.read();
	if (value) buf += decoder.decode(value, { stream: true });
	let nl = buf.indexOf("\\n");
	while (nl !== -1) {
		const line = buf.slice(0, nl).trim();
		buf = buf.slice(nl + 1);
		if (line) {
			const parsed = JSON.parse(line);
			await handleTurn(parsed?.message?.content ?? "");
		}
		nl = buf.indexOf("\\n");
	}
	if (done) break;
}
`,
			"fake-agy-stream-followup",
		);

		let sentFollowUp = false;
		let followUpReceipt: Awaited<ReturnType<typeof IrcBus.prototype.send>> | undefined;
		const runPromise = runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Initial assignment",
			index: 0,
			id: "StreamFollowUpWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: false,
			settings: makeSettings(streamBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!sentFollowUp && p.currentTool === "read_file") {
					sentFollowUp = true;
					// Sibling message (non-parent) queues as a next-turn follow-up on live stdin without killing Turn 1.
					void IrcBus.global()
						.send({ from: "SiblingWorker", to: "StreamFollowUpWorker", body: "Also check bar.ts" })
						.then(r => {
							followUpReceipt = r;
						});
				}
			},
		});

		const res = await runPromise;
		expect(res.exitCode).toBe(0);
		expect(followUpReceipt?.outcome).toBe("injected");
		expect(res.output).toContain("reply-turn-2");
		expect(res.output).toContain("Also check bar.ts");
		expect(res.requests).toBe(2);

		// Both turns ran inside the exact same live agy process (same PID, no restart).
		const logged = (await fs.readFile(logFile, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as { pid: number; turn: number; content: string });
		expect(logged).toHaveLength(2);
		expect(logged[0]?.pid).toBe(logged[1]?.pid);
		expect(logged[0]?.turn).toBe(1);
		expect(logged[1]?.turn).toBe(2);
		expect(logged[1]?.content).toContain("SiblingWorker");
		expect(logged[1]?.content).toContain("Also check bar.ts");

		// Transcript preserves intact parentId chain and records the queued follow-up prompt between turns.
		const entries = (await fs.readFile(path.join(artifactsDir, "StreamFollowUpWorker.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		let prevId: string | null = null;
		for (const entry of entries.slice(1)) {
			expect(entry.parentId).toBe(prevId);
			prevId = entry.id as string;
		}
		expect(
			entries.some(
				e =>
					e.type === "message" &&
					(e.message as Record<string, unknown>).role === "user" &&
					JSON.stringify(e.message).includes("Also check bar.ts"),
			),
		).toBe(true);
	}, 30_000);

	it("interrupts an active agy turn on mid-job parent IRC steer and RPC steer_subagent, then resumes via --conversation", async () => {
		const dir = await makeTempDir("omp-agy-midjob-steer-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-st" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });
		const logFile = path.join(dir, "steer-invocations.jsonl");

		// First invocation sleeps for 15s (simulating a long turn) so an immediate steer MUST interrupt it;
		// resumed invocation (--conversation conv-steer-1) completes immediately with the steered answer.
		const steerBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
fs.appendFileSync(${JSON.stringify(logFile)}, JSON.stringify({ pid: process.pid, conv, promptText }) + "\\n");
const cid = conv ?? "conv-steer-1";
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
if (!conv) {
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "run_command", tool_info: { parameters: { CommandLine: "sleep 15" } } } });
	await Bun.sleep(15_000);
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "unsteered-slow-finish", num_turns: 1 } });
} else {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "STEERED_OK:" + promptText, num_turns: 2, usage: { input_tokens: 150, output_tokens: 40, cache_read_tokens: 80 } } });
}
`,
			"fake-agy-steer",
		);

		// 1. Direct parent IRC steer interrupts long turn immediately and resumes via --conversation.
		let sentParentSteer = false;
		const t0 = Date.now();
		const parentSteerRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Long initial task",
			index: 0,
			id: "ParentSteerWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: false,
			settings: makeSettings(steerBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!sentParentSteer && p.currentTool === "run_command") {
					sentParentSteer = true;
					void IrcBus.global().send({
						from: MAIN_AGENT_ID,
						to: "ParentSteerWorker",
						body: "Stop and switch to fix-B instead",
					});
				}
			},
		});
		const elapsedParentSteer = Date.now() - t0;
		expect(parentSteerRes.exitCode).toBe(0);
		expect(parentSteerRes.aborted).toBeUndefined();
		expect(parentSteerRes.output).toContain("STEERED_OK:");
		expect(parentSteerRes.output).toContain("Stop and switch to fix-B instead");
		// Interrupted long 15s turn promptly rather than waiting for Turn 1 to finish.
		expect(elapsedParentSteer).toBeLessThan(5_000);

		const parentCalls = (await fs.readFile(logFile, "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as { pid: number; conv: string | null; promptText: string });
		expect(parentCalls).toHaveLength(2);
		expect(parentCalls[0]?.conv).toBeNull();
		expect(parentCalls[1]?.conv).toBe("conv-steer-1");
		expect(parentCalls[1]?.promptText).toContain("Stop and switch to fix-B instead");

		// 2. Herdrup/RPC handleRpcSteerSubagent resolves live agy subagent and steers it mid-turn.
		const rpcBus = new EventBus();
		const rpcSubagents = new RpcSubagentRegistry(rpcBus, () => {});
		let rpcSteerErr: string | undefined = "not-called";
		let rpcSteerSent = false;
		const rpcSteerRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Long RPC task",
			index: 0,
			id: "RpcSteerWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			eventBus: rpcBus,
			keepAlive: false,
			settings: makeSettings(steerBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!rpcSteerSent && p.currentTool === "run_command") {
					rpcSteerSent = true;
					void handleRpcSteerSubagent(rpcSubagents, "RpcSteerWorker", "Pivot from Herdrup UI").then(err => {
						rpcSteerErr = err;
					});
				}
			},
		});
		rpcSubagents.dispose();
		expect(rpcSteerErr).toBeUndefined();
		expect(rpcSteerRes.exitCode).toBe(0);
		expect(rpcSteerRes.output).toContain("STEERED_OK:Pivot from Herdrup UI");
	}, 30_000);

	it("cancels a running agy subagent cleanly via RPC cancel_subagent and preserves its aborted tombstone", async () => {
		const dir = await makeTempDir("omp-agy-rpc-cancel-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-cancel" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const slowBin = await createFakeAgyBinary(
			dir,
			`
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: "conv-cancel-1", init: { model: "gemini-4-neon-low" } });
emit({ event: "step_update", step_update: { conversation_id: "conv-cancel-1", step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "bash", tool_info: { parameters: { command: "sleep 30" } } } });
await Bun.sleep(30_000);
`,
			"fake-agy-slow-cancel",
		);

		const rpcBus = new EventBus();
		const rpcSubagents = new RpcSubagentRegistry(rpcBus, () => {});
		let cancelTriggered = false;
		let cancelResult = false;
		const t0 = Date.now();
		const res = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Cancel me mid-run",
			index: 0,
			id: "CancelAgyWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			eventBus: rpcBus,
			keepAlive: true,
			settings: makeSettings(slowBin, { task: "gemini-4-neon-low" }),
			onProgress: p => {
				if (!cancelTriggered && p.currentTool === "bash") {
					cancelTriggered = true;
					void handleRpcCancelSubagent(rpcSubagents, "CancelAgyWorker").then(ok => {
						cancelResult = ok;
					});
				}
			},
		});
		rpcSubagents.dispose();

		expect(Date.now() - t0).toBeLessThan(5_000);
		expect(cancelResult).toBe(true);
		expect(res.exitCode).toBe(1);
		expect(res.aborted).toBe(true);
		const ref = AgentRegistry.global().get("CancelAgyWorker");
		expect(ref?.status).toBe("aborted");
		expect(ref?.session).toBeNull();
	}, 30_000);

	it("adopts keepAlive agy subagents for idle IRC wake turns, post-park AgentLifecycleManager revival, and deferred onRelease", async () => {
		const dir = await makeTempDir("omp-agy-revive-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-rev" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });
		const logFile = path.join(dir, "revive-calls.jsonl");

		const reviveBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const cid = conv ?? "conv-revive-42";
fs.appendFileSync(${JSON.stringify(logFile)}, JSON.stringify({ conv, promptText }) + "\\n");
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
emit({
	event: "result",
	result: {
		conversation_id: cid,
		status: "SUCCESS",
		response: (conv ? "RESUMED(" + conv + "):" : "INITIAL:") + promptText.slice(-40),
		num_turns: 1,
		usage: { input_tokens: 120, output_tokens: 30, cache_read_tokens: 60 },
	},
});
`,
			"fake-agy-revive",
		);

		// Register a lightweight main recipient in AgentRegistry so wakeRelay replies land cleanly via IrcBus.
		const receivedRelays: Array<{ from: string; body: string; wakeRelay?: boolean }> = [];
		const mainSessionStub = {
			deliverIrcMessage: async (msg: { from: string; body: string; wakeRelay?: boolean }) => {
				receivedRelays.push({ from: msg.from, body: msg.body, wakeRelay: msg.wakeRelay });
				return "injected" as const;
			},
			emitIrcRelayObservation: () => {},
		};
		AgentRegistry.global().register({
			id: MAIN_AGENT_ID,
			displayName: MAIN_AGENT_ID,
			kind: "main",
			session: mainSessionStub as unknown as import("../../src/session/agent-session-handle").AgentSessionHandle,
			status: "idle",
		});

		let releaseCount = 0;
		const initialRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Initial keepAlive work",
			index: 0,
			id: "KeepAliveAgy",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: true,
			onRelease: async () => {
				releaseCount += 1;
			},
			settings: makeSettings(reviveBin, { task: "gemini-4-neon-medium" }),
		});
		expect(initialRes.exitCode).toBe(0);
		expect(initialRes.output).toContain("INITIAL:");
		expect(releaseCount).toBe(0);

		// 1. Adopted into AgentLifecycleManager as idle with live handle (never mistaken for concrete AgentSession).
		const idleRef = AgentRegistry.global().get("KeepAliveAgy");
		expect(idleRef?.status).toBe("idle");
		expect(idleRef?.session).not.toBeNull();
		expect(isLocalAgentSession(idleRef!.session!)).toBe(false);
		expect(AgentLifecycleManager.global().has("KeepAliveAgy")).toBe(true);

		// 2. Sending an IRC message wakes the idle agy subagent and relays the reply back to main.
		const wakeReceipt = await IrcBus.global().send({
			from: MAIN_AGENT_ID,
			to: "KeepAliveAgy",
			body: "Wake question 1",
		});
		expect(wakeReceipt.outcome).toBe("woken");
		await idleRef!.session!.waitForIrcReplies();
		expect(receivedRelays).toHaveLength(1);
		expect(receivedRelays[0]?.from).toBe("KeepAliveAgy");
		expect(receivedRelays[0]?.wakeRelay).toBe(true);
		expect(receivedRelays[0]?.body).toContain("RESUMED(conv-revive-42):");
		expect(receivedRelays[0]?.body).toContain("Wake question 1");
		expect(AgentRegistry.global().get("KeepAliveAgy")?.status).toBe("idle");

		// 3. Park the agent (disposes live handle), then message it again to exercise AgentReviver cold resume.
		await AgentLifecycleManager.global().park("KeepAliveAgy");
		const parkedRef = AgentRegistry.global().get("KeepAliveAgy");
		expect(parkedRef?.status).toBe("parked");
		expect(parkedRef?.session).toBeNull();
		expect(releaseCount).toBe(0);

		const reviveReceipt = await IrcBus.global().send({
			from: MAIN_AGENT_ID,
			to: "KeepAliveAgy",
			body: "Revive question 2",
		});
		expect(reviveReceipt.outcome).toBe("revived");
		const revivedSession = AgentRegistry.global().get("KeepAliveAgy")?.session;
		expect(revivedSession).not.toBeNull();
		await revivedSession!.waitForIrcReplies();
		expect(receivedRelays).toHaveLength(2);
		expect(receivedRelays[1]?.wakeRelay).toBe(true);
		expect(receivedRelays[1]?.body).toContain("RESUMED(conv-revive-42):");
		expect(receivedRelays[1]?.body).toContain("Revive question 2");

		// 4. Explicit runSubagentFollowUpTurn also works cleanly on adopted agy subagents.
		const followUpTurn = await runSubagentFollowUpTurn({
			id: "KeepAliveAgy",
			agent: makeAgent(),
			message: "Follow-up turn 3",
		});
		expect(followUpTurn.exitCode).toBe(0);
		expect(followUpTurn.output).toContain("RESUMED(conv-revive-42):");
		expect(followUpTurn.output).toContain("Follow-up turn 3");

		// 5. Transcript preserved a single continuous parentId chain across initial + wake + revived + follow-up turns.
		const transcriptLines = (await fs.readFile(path.join(artifactsDir, "KeepAliveAgy.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		let prevId: string | null = null;
		for (const entry of transcriptLines.slice(1)) {
			expect(entry.parentId).toBe(prevId);
			prevId = entry.id as string;
		}
		expect(JSON.stringify(transcriptLines)).toContain("Wake question 1");
		expect(JSON.stringify(transcriptLines)).toContain("Revive question 2");
		expect(JSON.stringify(transcriptLines)).toContain("Follow-up turn 3");

		// 6. Final release runs deferred onRelease exactly once.
		await AgentLifecycleManager.global().release("KeepAliveAgy");
		expect(releaseCount).toBe(1);
		expect(AgentRegistry.global().get("KeepAliveAgy")).toBeUndefined();
	}, 30_000);

	it("recovers acknowledged follow-ups when child closes stdin early and steers on step_update conversation ID with SIGINT", async () => {
		const dir = await makeTempDir("omp-agy-p2-stdin-sigint-");
		const checkpointFile = path.join(dir, "sigint-checkpoint.txt");

		// 1. Child closes fd 0 right after reading prompt, emits Turn 1 SUCCESS, and exits 0;
		// sibling follow-up accepted after fd 0 closes must still run via --conversation resume.
		const closeStdinBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
const cid = conv ?? "conv-closed-fd0";
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
if (!conv) {
	try { fs.closeSync(0); } catch {}
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "read_file", tool_info: { parameters: { Path: "a.ts" } } } });
	await Bun.sleep(250);
	emit({ event: "step_update", step_update: { conversation_id: cid, step_index: 1, state: "DONE", step_type: "tool", tool_name: "read_file", tool_info: { output: "ok" } } });
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "turn1-only", num_turns: 1 } });
} else {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "RECOVERED_FOLLOWUP:" + promptText, num_turns: 1 } });
}
`,
			"fake-agy-close-fd0",
		);

		let sentAfterClose = false;
		let receipt: Awaited<ReturnType<typeof IrcBus.prototype.send>> | undefined;
		const closedArtifactsDir = path.join(dir, "closed-artifacts");
		await fs.mkdir(closedArtifactsDir, { recursive: true });
		const recoveredRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Close stdin early",
			index: 0,
			id: "ClosedFd0Worker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir: closedArtifactsDir,
			keepAlive: false,
			settings: makeSettings(closeStdinBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!sentAfterClose && p.currentTool === "read_file") {
					sentAfterClose = true;
					void IrcBus.global()
						.send({ from: "SiblingPeer", to: "ClosedFd0Worker", body: "Must not be lost" })
						.then(r => {
							receipt = r;
						});
				}
			},
		});
		expect(receipt?.outcome).toBe("injected");
		expect(recoveredRes.exitCode).toBe(0);
		expect(recoveredRes.output).toContain("RECOVERED_FOLLOWUP:");
		expect(recoveredRes.output).toContain("Must not be lost");
		// Closed-stdin recovery must record the Turn 1 reply and the recovered prompt exactly ONCE each in JSONL.
		const closedTranscriptLines = (await fs.readFile(path.join(closedArtifactsDir, "ClosedFd0Worker.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		const turn1ReplyEntries = closedTranscriptLines.filter(
			e =>
				e.type === "message" &&
				(e.message as Record<string, unknown>).role === "assistant" &&
				JSON.stringify(e.message).includes("turn1-only"),
		);
		const recoveredPromptEntries = closedTranscriptLines.filter(
			e =>
				e.type === "message" &&
				(e.message as Record<string, unknown>).role === "user" &&
				JSON.stringify(e.message).includes("Must not be lost"),
		);
		expect(turn1ReplyEntries).toHaveLength(1);
		expect(recoveredPromptEntries).toHaveLength(1);

		// 2. Steer queued BEFORE conversation_id is emitted on step_update (not init) must interrupt promptly
		// AND deliver SIGINT (not SIGTERM) so the child's SIGINT checkpoint handler runs before resume.
		const sigintBin = await createFakeAgyBinary(
			dir,
			`
import * as fs from "node:fs";
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
process.on("SIGINT", () => {
	fs.writeFileSync(${JSON.stringify(checkpointFile)}, "SIGINT");
	process.exit(130);
});
process.on("SIGTERM", () => {
	fs.writeFileSync(${JSON.stringify(checkpointFile)}, "SIGTERM");
	process.exit(143);
});
if (!conv) {
	// Emit init WITHOUT conversation_id, wait briefly so steer queues before ID arrives, then emit ID on step_update.
	emit({ event: "init", init: { model: "gemini-4-neon-medium" } });
	emit({ event: "step_update", step_update: { step_index: 1, state: "ACTIVE", step_type: "tool", tool_name: "prep", tool_info: { parameters: { Path: "p" } } } });
	await Bun.sleep(150);
	emit({ event: "step_update", step_update: { conversation_id: "conv-late-id", step_index: 2, state: "ACTIVE", step_type: "tool", tool_name: "slow_work", tool_info: { parameters: { Path: "s" } } } });
	await Bun.sleep(10_000);
	emit({ event: "result", result: { conversation_id: "conv-late-id", status: "SUCCESS", response: "missed-steer", num_turns: 1 } });
} else {
	const sig = fs.existsSync(${JSON.stringify(checkpointFile)}) ? fs.readFileSync(${JSON.stringify(checkpointFile)}, "utf8") : "NONE";
	emit({ event: "init", conversation_id: conv, init: { model: "gemini-4-neon-medium" } });
	emit({ event: "result", result: { conversation_id: conv, status: "SUCCESS", response: "SIGNAL=" + sig + ":" + promptText, num_turns: 1 } });
}
`,
			"fake-agy-sigint-late-id",
		);

		let steerQueuedEarly = false;
		const tSteer = Date.now();
		const steerRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Late ID + SIGINT test",
			index: 0,
			id: "LateIdSigintWorker",
			parentAgentId: MAIN_AGENT_ID,
			keepAlive: false,
			settings: makeSettings(sigintBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!steerQueuedEarly && p.currentTool === "prep") {
					steerQueuedEarly = true;
					void IrcBus.global().send({
						from: MAIN_AGENT_ID,
						to: "LateIdSigintWorker",
						body: "Pivot immediately",
					});
				}
			},
		});
		expect(Date.now() - tSteer).toBeLessThan(4_000);
		expect(steerRes.exitCode).toBe(0);
		expect(steerRes.output).toContain("SIGNAL=SIGINT:");
		expect(steerRes.output).toContain("Pivot immediately");
	}, 30_000);

	it("closes one-shot admission and commits terminal state atomically before terminal observers, and pumps agent_end wake messages", async () => {
		const dir = await makeTempDir("omp-agy-p2-teardown-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-td" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const fastBin = await createFakeAgyBinary(
			dir,
			`
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const cid = conv ?? "conv-td-1";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-low" } });
emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "OK:" + promptText.slice(-30), num_turns: 1 } });
`,
			"fake-agy-teardown",
		);

		// 1. One-shot run: sibling send and RPC cancel from terminal onProgress must both fail/return false, never conflict.
		const rpcBus = new EventBus();
		const rpcSubagents = new RpcSubagentRegistry(rpcBus, () => {});
		let terminalSendOutcome: string | undefined;
		let terminalCancelResult: boolean | undefined;
		let ranTerminalHooks = false;
		const oneShotRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "One-shot teardown atomicity",
			index: 0,
			id: "OneShotAtomicWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			eventBus: rpcBus,
			keepAlive: false,
			settings: makeSettings(fastBin, { task: "gemini-4-neon-low" }),
			onProgress: p => {
				if (p.status === "completed" && !ranTerminalHooks) {
					ranTerminalHooks = true;
					void IrcBus.global()
						.send({ from: "Sibling", to: "OneShotAtomicWorker", body: "too late" })
						.then(r => {
							terminalSendOutcome = r.outcome;
						});
					void handleRpcCancelSubagent(rpcSubagents, "OneShotAtomicWorker").then(ok => {
						terminalCancelResult = ok;
					});
				}
			},
		});
		await Bun.sleep(30);
		rpcSubagents.dispose();
		expect(oneShotRes.exitCode).toBe(0);
		expect(terminalSendOutcome).toBe("failed");
		expect(terminalCancelResult).toBe(false);

		// 2. Kept-alive wake: message delivered synchronously inside agent_end must run (not strand ircWakesPending=true).
		const relays: string[] = [];
		AgentRegistry.global().register({
			id: MAIN_AGENT_ID,
			displayName: MAIN_AGENT_ID,
			kind: "main",
			session: {
				deliverIrcMessage: async (msg: { body: string }) => {
					relays.push(msg.body);
					return "injected" as const;
				},
				emitIrcRelayObservation: () => {},
			} as unknown as import("../../src/session/agent-session-handle").AgentSessionHandle,
			status: "idle",
		});

		const kaRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "KeepAlive agent_end wake test",
			index: 0,
			id: "AgentEndWakeWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: true,
			settings: makeSettings(fastBin, { task: "gemini-4-neon-low" }),
		});
		expect(kaRes.exitCode).toBe(0);
		const liveSession = AgentRegistry.global().get("AgentEndWakeWorker")!.session!;
		let sentDuringAgentEnd = false;
		let agentEndReceipt: Awaited<ReturnType<typeof IrcBus.prototype.send>> | undefined;
		const unsubEnd = liveSession.subscribe(ev => {
			if (ev.type === "agent_end" && !sentDuringAgentEnd) {
				sentDuringAgentEnd = true;
				void IrcBus.global()
					.send({ from: MAIN_AGENT_ID, to: "AgentEndWakeWorker", body: "chained-from-agent-end" })
					.then(r => {
						agentEndReceipt = r;
					});
			}
		});
		await IrcBus.global().send({ from: MAIN_AGENT_ID, to: "AgentEndWakeWorker", body: "first-wake" });
		await liveSession.waitForIrcReplies();
		await Bun.sleep(50);
		await liveSession.waitForIrcReplies();
		unsubEnd();

		expect(agentEndReceipt?.outcome).toBe("woken");
		expect(relays.some(r => r.includes("first-wake"))).toBe(true);
		expect(relays.some(r => r.includes("chained-from-agent-end"))).toBe(true);
		expect(liveSession.ircWakesPending).toBe(false);
		expect(AgentRegistry.global().get("AgentEndWakeWorker")?.status).toBe("idle");
	}, 30_000);

	it("serializes concurrent monitored follow-ups, enforces their schema/observer/artifact contract, and bridges agy refs across worker RPC", async () => {
		const dir = await makeTempDir("omp-agy-p2-fu-bridge-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-fu2" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const followUpBin = await createFakeAgyBinary(
			dir,
			`
${READ_FIRST_STDIN_PROMPT_SNIPPET}
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const promptText = await readFirstPrompt();
const cid = conv ?? "conv-fu-contract";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
await Bun.sleep(120);
if (promptText.includes("PROSE_BAD")) {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "plain prose", num_turns: 1 } });
} else if (promptText.includes("JSON_GOOD")) {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: '{"answer":"structured-ok"}', structured_output: { answer: "structured-ok" }, num_turns: 1 } });
} else {
	emit({ event: "result", result: { conversation_id: cid, status: "SUCCESS", response: "DONE:" + promptText, num_turns: 1 } });
}
`,
			"fake-agy-fu-contract",
		);

		const initRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Initial run",
			index: 0,
			id: "ContractAgyWorker",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: true,
			settings: makeSettings(followUpBin, { task: "gemini-4-neon-medium" }),
		});
		expect(initRes.exitCode).toBe(0);

		// 1. Concurrent runSubagentFollowUpTurn calls serialize without corrupting <id>.jsonl entry IDs / parentId chain.
		const [fuA, fuB] = await Promise.all([
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "concurrent-A" }),
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "concurrent-B" }),
		]);
		expect(fuA.exitCode).toBe(0);
		expect(fuB.exitCode).toBe(0);
		const lines = (await fs.readFile(path.join(artifactsDir, "ContractAgyWorker.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(l => JSON.parse(l) as Record<string, unknown>);
		const ids = new Set<string>();
		let prevId: string | null = null;
		for (const entry of lines.slice(1)) {
			const eid = entry.id as string;
			expect(ids.has(eid)).toBe(false);
			ids.add(eid);
			expect(entry.parentId).toBe(prevId);
			prevId = eid;
		}

		// 2. Monitored follow-up enforces outputSchema, emits progress/lifecycle events, and updates artifactsDir on success.
		const fuBus = new EventBus();
		const fuLifecycles: string[] = [];
		const fuProgress: string[] = [];
		fuBus.on(TASK_SUBAGENT_LIFECYCLE_CHANNEL, p => fuLifecycles.push((p as SubagentLifecyclePayload).status));
		const schema = { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] };
		const badFu = await runSubagentFollowUpTurn({
			id: "ContractAgyWorker",
			agent: makeAgent(),
			message: "PROSE_BAD",
			outputSchema: schema,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			artifactsDir,
			eventBus: fuBus,
			onProgress: p => fuProgress.push(p.status),
		});
		expect(badFu.exitCode).toBe(1);
		expect(badFu.structuredOutput?.status).toBe("invalid");
		expect(fuLifecycles).toEqual(["started", "failed"]);
		expect(fuProgress.at(-1)).toBe("failed");

		const goodFu = await runSubagentFollowUpTurn({
			id: "ContractAgyWorker",
			agent: makeAgent(),
			message: "JSON_GOOD",
			outputSchema: schema,
			outputSchemaMode: "strict",
			outputSchemaSource: "caller",
			artifactsDir,
		});
		expect(goodFu.exitCode).toBe(0);
		expect(goodFu.structuredOutput).toEqual({
			source: "caller",
			mode: "strict",
			status: "valid",
			data: { answer: "structured-ok" },
		});
		expect(JSON.parse(await fs.readFile(path.join(artifactsDir, "ContractAgyWorker.json"), "utf8"))).toEqual({
			answer: "structured-ok",
		});

		// 3. Incompatible native-OMP-only follow-up options fail explicitly instead of silently ignoring them.
		await expect(
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "x", restore: true }),
		).rejects.toThrow(/restore/i);
		await expect(
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "x", topicBoard: "board" }),
		).rejects.toThrow(/topicBoard/i);
		await expect(
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "x", completionProbe: true }),
		).rejects.toThrow(/completionProbe/i);
		await expect(
			runSubagentFollowUpTurn({ id: "ContractAgyWorker", agent: makeAgent(), message: "x", softRequestBudget: 1 }),
		).rejects.toThrow(/softRequestBudget/i);

		// 3b. RPC cancel from a monitored follow-up's terminal onProgress must return false (committed idle first).
		const fuRpcBus = new EventBus();
		const fuRpcSubagents = new RpcSubagentRegistry(fuRpcBus, () => {});
		let fuTerminalCancel: boolean | undefined;
		let fuCancelChecked = false;
		const fuCancelRes = await runSubagentFollowUpTurn({
			id: "ContractAgyWorker",
			agent: makeAgent(),
			message: "check-monitored-cancel-atomicity",
			eventBus: fuRpcBus,
			onProgress: p => {
				if (p.status === "completed" && !fuCancelChecked) {
					fuCancelChecked = true;
					void handleRpcCancelSubagent(fuRpcSubagents, "ContractAgyWorker").then(ok => {
						fuTerminalCancel = ok;
					});
				}
			},
		});
		await Bun.sleep(20);
		fuRpcSubagents.dispose();
		expect(fuCancelRes.exitCode).toBe(0);
		expect(fuTerminalCancel).toBe(false);
		expect(AgentRegistry.global().get("ContractAgyWorker")?.status).toBe("idle");

		// 3c. A monitored follow-up triggered the instant the initial run flips idle must wait on the shared turn lock
		// so it never corrupts the still-flushing initial transcript or gets overwritten by the initial artifact.
		let earlyFollowUpPromise: Promise<import("../../src/task/task-details").SingleResult> | undefined;
		const unsubRace = AgentRegistry.global().onChange(ev => {
			if (
				ev.type === "status_changed" &&
				ev.ref.id === "InitialLockRaceWorker" &&
				ev.ref.status === "idle" &&
				!earlyFollowUpPromise
			) {
				earlyFollowUpPromise = runSubagentFollowUpTurn({
					id: "InitialLockRaceWorker",
					agent: makeAgent(),
					message: "FOLLOWUP_AFTER_IDLE_EVENT",
					artifactsDir,
				});
			}
		});
		try {
			const raceInitRes = await runSubprocess({
				cwd: dir,
				agent: makeAgent(),
				task: "INITIAL_LOCK_RACE",
				index: 0,
				id: "InitialLockRaceWorker",
				parentAgentId: MAIN_AGENT_ID,
				artifactsDir,
				sessionFile: parentSession,
				keepAlive: true,
				settings: makeSettings(followUpBin, { task: "gemini-4-neon-medium" }),
			});
			expect(raceInitRes.exitCode).toBe(0);
			expect(earlyFollowUpPromise).toBeDefined();
			const raceFuRes = await earlyFollowUpPromise!;
			expect(raceFuRes.exitCode).toBe(0);
			// Final artifact must hold FOLLOWUP_AFTER_IDLE_EVENT (never overwritten by INITIAL_LOCK_RACE).
			const latestArtifact = await fs.readFile(path.join(artifactsDir, "InitialLockRaceWorker.md"), "utf8");
			expect(latestArtifact).toContain("FOLLOWUP_AFTER_IDLE_EVENT");
			// Transcript must be valid JSONL with unique sequential IDs and an intact parentId chain.
			const raceLines = (await fs.readFile(path.join(artifactsDir, "InitialLockRaceWorker.jsonl"), "utf8"))
				.trim()
				.split("\n")
				.map(l => JSON.parse(l) as Record<string, unknown>);
			const raceIds = new Set<string>();
			let racePrev: string | null = null;
			for (const entry of raceLines.slice(1)) {
				const eid = entry.id as string;
				expect(raceIds.has(eid)).toBe(false);
				raceIds.add(eid);
				expect(entry.parentId).toBe(racePrev);
				racePrev = eid;
			}
		} finally {
			unsubRace();
		}

		// 4. Real worker calling AgentRegistry.global().get() for a main-homed agy agent crosses the bridge cleanly.
		const probeReady = Promise.withResolvers<AgentsProbe>();
		const probeService: ThreadService = {
			name: "agents-probe",
			globals: (): AgentsProbeGlobals => ({ ready: received => probeReady.resolve(received) }),
		};
		const worker = new SubagentWorker({
			entry: new URL("../fixtures/agents-bridge-worker.ts", import.meta.url),
			services: [identityThreadService, agentsThreadService, probeService],
			initTimeoutMs: 60_000,
		});
		try {
			await worker.ready;
			const probe = await probeReady.promise;
			const peerReport = await probe.peer("ContractAgyWorker");
			expect(peerReport.isPeer).toBe(true);
			expect(peerReport.key.startsWith("agy:")).toBe(true);
			expect(peerReport.isStreaming).toBe(false);
			expect(peerReport.sessionFile).toBe(path.join(artifactsDir, "ContractAgyWorker.jsonl"));
			expect(peerReport.canonical).toBe(true);
			expect(peerReport.casBySession).toBe(true);
		} finally {
			await worker.terminate("test end");
		}
	}, 30_000);

	it("does not duplicate prior answer or follow-up prompt when successor emits ACTIVE then exits 0 without a second result", async () => {
		const dir = await makeTempDir("omp-agy-successor-exit0-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-s0" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const successorDropBin = await createFakeAgyBinary(
			dir,
			`
import * as readline from "node:readline";
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const cid = conv ?? "conv-successor-drop";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const linesIter = rl[Symbol.asyncIterator]();
const readPromptLine = async () => {
	while (true) {
		const next = await linesIter.next();
		if (next.done) return null;
		if (next.value.trim()) return JSON.parse(next.value)?.message?.content ?? "";
	}
};

const firstContent = await readPromptLine();
if (firstContent !== null) {
	if (!conv) {
		// Keep the tool ACTIVE until the follow-up arrives on live stdin (event-driven, no timer race).
		emit({
			event: "step_update",
			step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "ACTIVE", tool_info: { name: "bash", parameters: { command: "wait" } } },
		});
		const secondContent = await readPromptLine();
		emit({
			event: "step_update",
			step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "DONE", tool_info: { name: "bash", output: "ok" } },
		});
		emit({
			event: "result",
			result: {
				conversation_id: cid,
				status: "SUCCESS",
				response: "UNIQUE_FIRST_ANSWER_841",
				num_turns: 1,
				usage: { input_tokens: 40, output_tokens: 10, cache_read_tokens: 0 },
			},
		});
		if (secondContent !== null) {
			// Emit one valid successor ACTIVE frame, then exit 0 WITHOUT emitting a second result frame.
			emit({
				event: "step_update",
				step_update: { step_index: 2, step_type: "agent_response", state: "ACTIVE" },
			});
			process.exit(0);
		}
	} else {
		emit({
			event: "result",
			result: {
				conversation_id: cid,
				status: "SUCCESS",
				response: "UNIQUE_SECOND_ANSWER_842:" + firstContent.slice(-30),
				num_turns: 1,
				usage: { input_tokens: 50, output_tokens: 15, cache_read_tokens: 0 },
			},
		});
	}
}
`,
			"fake-agy-successor-drop",
		);

		let injected = false;
		const res = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Initial work before successor drop",
			index: 0,
			id: "SuccessorDropAgy",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: false,
			settings: makeSettings(successorDropBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!injected && p.currentTool === "bash") {
					injected = true;
					void IrcBus.global().send({
						from: "PeerWorker",
						to: "SuccessorDropAgy",
						body: "UNIQUE_FOLLOWUP_PROMPT_841",
					});
				}
			},
		});

		expect(res.exitCode).toBe(0);
		expect(res.output).toContain("UNIQUE_SECOND_ANSWER_842:");
		expect(res.output).toContain("UNIQUE_FOLLOWUP_PROMPT_841");

		const transcriptLines = (await fs.readFile(path.join(artifactsDir, "SuccessorDropAgy.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(
				l =>
					JSON.parse(l) as {
						type: string;
						message?: {
							role?: string;
							content?: Array<{ text?: string }>;
							usage?: { input?: number; output?: number };
						};
					},
			);

		const assistantEntries = transcriptLines.filter(e => e.type === "message" && e.message?.role === "assistant");
		const assistantTexts = assistantEntries.map(e => (e.message?.content ?? []).map(c => c.text ?? "").join(""));
		const userTexts = transcriptLines
			.filter(e => e.type === "message" && e.message?.role === "user")
			.map(e => (e.message?.content ?? []).map(c => c.text ?? "").join(""));

		expect(assistantTexts.filter(t => t.includes("UNIQUE_FIRST_ANSWER_841"))).toHaveLength(1);
		expect(userTexts.filter(t => t.includes("UNIQUE_FOLLOWUP_PROMPT_841"))).toHaveLength(1);
		expect(assistantTexts.filter(t => t.includes("UNIQUE_SECOND_ANSWER_842:"))).toHaveLength(1);

		// Transcript accounting must preserve result-only usage from Turn 1 (40 in / 10 out) plus resumed Turn 2 (50 in / 15 out).
		const totalTranscriptInput = assistantEntries.reduce((sum, e) => sum + (e.message?.usage?.input ?? 0), 0);
		const totalTranscriptOutput = assistantEntries.reduce((sum, e) => sum + (e.message?.usage?.output ?? 0), 0);
		expect(totalTranscriptInput).toBe(90);
		expect(totalTranscriptOutput).toBe(25);
		expect(res.usage?.input).toBe(90);
		expect(res.usage?.output).toBe(25);
	}, 30_000);

	it("reserves the initial turn lock before exposing the session on registered and commits monitored idle once", async () => {
		const dir = await makeTempDir("omp-agy-startup-lock-single-idle-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-sl" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const bin = await createFakeAgyBinary(
			dir,
			`
import * as readline from "node:readline";
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const conv = convIdx >= 0 ? argv[convIdx + 1] : null;
const cid = conv ?? "conv-startup-lock";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
let lastContent = "";
let turns = 0;
for await (const line of rl) {
	if (!line.trim()) continue;
	const msg = JSON.parse(line);
	lastContent = msg?.message?.content ?? "";
	turns += 1;
	await new Promise(r => setTimeout(r, 30));
	emit({
		event: "result",
		result: {
			conversation_id: cid,
			status: "SUCCESS",
			response: "ECHO:" + lastContent.slice(-60),
			num_turns: turns,
			usage: { input_tokens: 20, output_tokens: 10, cache_read_tokens: 0 },
		},
	});
}
`,
			"fake-agy-startup-lock",
		);

		// Park an IRC message before the subagent starts so startup hits the async parked-message drain.
		IrcBus.global().park({
			from: "PeerStartup",
			to: "StartupLockWorker",
			body: "PARKED_BEFORE_START_901",
		});

		let monitoredFromRegistered: Promise<import("../../src/task/task-details").SingleResult> | undefined;
		const unsubReg = AgentRegistry.global().onChange(ev => {
			if (ev.type === "registered" && ev.ref.id === "StartupLockWorker" && !monitoredFromRegistered) {
				monitoredFromRegistered = runSubagentFollowUpTurn({
					id: "StartupLockWorker",
					agent: makeAgent(),
					message: "MONITORED_FROM_REGISTERED_902",
					artifactsDir,
				});
			}
		});

		try {
			const initRes = await runSubprocess({
				cwd: dir,
				agent: makeAgent(),
				task: "INITIAL_TASK_900",
				index: 0,
				id: "StartupLockWorker",
				parentAgentId: MAIN_AGENT_ID,
				artifactsDir,
				sessionFile: parentSession,
				keepAlive: true,
				settings: makeSettings(bin, { task: "gemini-4-neon-medium" }),
			});

			expect(monitoredFromRegistered).toBeDefined();
			const monRes = await monitoredFromRegistered!;

			expect(initRes.exitCode).toBe(0);
			expect(monRes.exitCode).toBe(0);
			// Initial turn must consume the parked message; monitored turn must execute second and return its own answer.
			expect(initRes.output).toContain("PARKED_BEFORE_START_901");
			expect(monRes.output).toContain("MONITORED_FROM_REGISTERED_902");
			expect(monRes.output).not.toContain("PARKED_BEFORE_START_901");

			// Final artifact must hold the monitored turn's output (never overwritten by the initial turn).
			const artifactText = await fs.readFile(path.join(artifactsDir, "StartupLockWorker.md"), "utf8");
			expect(artifactText).toContain("MONITORED_FROM_REGISTERED_902");

			// Every JSONL line must parse cleanly with unique IDs and intact parentId chain.
			const rawLines = (await fs.readFile(path.join(artifactsDir, "StartupLockWorker.jsonl"), "utf8"))
				.trim()
				.split("\n");
			const parsed = rawLines.map(l => JSON.parse(l) as Record<string, unknown>);
			const seenIds = new Set<string>();
			let prev: string | null = null;
			for (const entry of parsed.slice(1)) {
				const eid = entry.id as string;
				expect(seenIds.has(eid)).toBe(false);
				seenIds.add(eid);
				expect(entry.parentId).toBe(prev);
				prev = eid;
			}
		} finally {
			unsubReg();
		}

		// Verify observable running->idle settlement and retained-session availability on both normal completion
		// and caller-aborted monitored turns (caller abort must return the retained ref to idle, not leave it running).
		const registry = AgentRegistry.global();
		const observedStatuses: string[] = [];
		const unsubStatus = registry.onChange(ev => {
			if (ev.type === "status_changed" && ev.ref.id === "StartupLockWorker") {
				observedStatuses.push(ev.ref.status);
			}
		});
		try {
			const secondMon = await runSubagentFollowUpTurn({
				id: "StartupLockWorker",
				agent: makeAgent(),
				message: "NORMAL_SETTLE_903",
				artifactsDir,
			});
			expect(secondMon.exitCode).toBe(0);
			expect(observedStatuses).toEqual(["running", "idle"]);
			const idleRefAfterNormal = registry.get("StartupLockWorker");
			expect(idleRefAfterNormal?.status).toBe("idle");
			expect(idleRefAfterNormal?.session?.runStartedAt).toBeUndefined();

			// Caller-aborted monitored turn must still transition the owned live ref back to idle and leave
			// the retained session usable for subsequent turns and lifecycle parking.
			observedStatuses.length = 0;
			const slowAbortBin = await createFakeAgyBinary(
				dir,
				`
import * as readline from "node:readline";
const argv = process.argv.slice(2);
const convIdx = argv.indexOf("--conversation");
const cid = convIdx >= 0 ? argv[convIdx + 1] : "conv-abort-idle";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
for await (const line of rl) {
	if (!line.trim()) continue;
	const msg = JSON.parse(line);
	const content = msg?.message?.content ?? "";
	if (content.includes("SLOW_CALLER_ABORT_904")) {
		emit({
			event: "step_update",
			step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "ACTIVE", tool_info: { name: "bash", parameters: { command: "sleep 10" } } },
		});
		await new Promise(r => setTimeout(r, 10_000));
	}
	emit({
		event: "result",
		result: {
			conversation_id: cid,
			status: "SUCCESS",
			response: "AFTER_ABORT_OK:" + content.slice(-30),
			num_turns: 1,
			usage: { input_tokens: 15, output_tokens: 5, cache_read_tokens: 0 },
		},
	});
	break;
}
`,
				"fake-agy-caller-abort",
			);

			const abortInit = await runSubprocess({
				cwd: dir,
				agent: makeAgent(),
				task: "ABORT_RETAINED_INIT",
				index: 0,
				id: "CallerAbortRetainedWorker",
				parentAgentId: MAIN_AGENT_ID,
				artifactsDir,
				sessionFile: parentSession,
				keepAlive: true,
				settings: makeSettings(slowAbortBin, { task: "gemini-4-neon-medium" }),
			});
			expect(abortInit.exitCode).toBe(0);

			const abortStatuses: string[] = [];
			const unsubAbortStatus = registry.onChange(ev => {
				if (ev.type === "status_changed" && ev.ref.id === "CallerAbortRetainedWorker") {
					abortStatuses.push(ev.ref.status);
				}
			});
			try {
				const callerAbort = new AbortController();
				const abortedTurn = await runSubagentFollowUpTurn({
					id: "CallerAbortRetainedWorker",
					agent: makeAgent(),
					message: "SLOW_CALLER_ABORT_904",
					signal: callerAbort.signal,
					onProgress: p => {
						if (p.currentTool === "bash" && !callerAbort.signal.aborted) {
							callerAbort.abort("Caller cancelled turn");
						}
					},
				});
				expect(abortedTurn.exitCode).toBe(1);
				expect(abortedTurn.aborted).toBe(true);
				expect(abortStatuses).toEqual(["running", "idle"]);

				const refAfterCallerAbort = registry.get("CallerAbortRetainedWorker");
				expect(refAfterCallerAbort?.status).toBe("idle");
				expect(refAfterCallerAbort?.session).not.toBeNull();
				expect(refAfterCallerAbort?.session?.runStartedAt).toBeUndefined();

				// A follow-up turn after caller abort still succeeds on the retained session.
				const recoveredTurn = await runSubagentFollowUpTurn({
					id: "CallerAbortRetainedWorker",
					agent: makeAgent(),
					message: "POST_ABORT_FOLLOWUP_905",
				});
				expect(recoveredTurn.exitCode).toBe(0);
				expect(recoveredTurn.output).toContain("AFTER_ABORT_OK:");
				expect(recoveredTurn.output).toContain("POST_ABORT_FOLLOWUP_905");
				expect(registry.get("CallerAbortRetainedWorker")?.status).toBe("idle");
			} finally {
				unsubAbortStatus();
			}
		} finally {
			unsubStatus();
		}
	}, 30_000);

	it("preserves prior reconciled usage on partial final frames and does not carry first-request context onto a result-only successor", async () => {
		const dir = await makeTempDir("omp-agy-partial-final-usage-");
		const parentSession = path.join(dir, "parent.jsonl");
		const artifactsDir = path.join(dir, "parent");
		await fs.writeFile(parentSession, `${JSON.stringify({ type: "session", version: 3, id: "p-pfu" })}\n`);
		await fs.mkdir(artifactsDir, { recursive: true });

		const partialUsageBin = await createFakeAgyBinary(
			dir,
			`
import * as readline from "node:readline";
const cid = "conv-partial-usage";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const linesIter = rl[Symbol.asyncIterator]();
const readPromptLine = async () => {
	while (true) {
		const next = await linesIter.next();
		if (next.done) return null;
		if (next.value.trim()) return JSON.parse(next.value)?.message?.content ?? "";
	}
};

const first = await readPromptLine();
if (first !== null) {
	// Keep tool ACTIVE until the live follow-up arrives on stdin (no timer race with progress coalescer).
	emit({
		event: "step_update",
		step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "ACTIVE", tool_info: { name: "bash", parameters: { command: "wait" } } },
	});
	const second = await readPromptLine();
	emit({
		event: "step_update",
		step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "DONE", tool_info: { name: "bash", output: "ok" } },
	});
	// Turn 1 reports usage only on its result frame (40 in / 10 out / 8 cache_read).
	emit({
		event: "result",
		result: {
			conversation_id: cid,
			status: "SUCCESS",
			response: "TURN1_USAGE_ANSWER",
			num_turns: 1,
			usage: { input_tokens: 40, output_tokens: 10, cache_read_tokens: 8 },
		},
	});
	if (second !== null) {
		// Successor reports its own DONE usage (50 in / 15 out / 12 cache_read), then its final SUCCESS frame
		// reports only cumulative input_tokens: 90 and omits output_tokens and cache_read_tokens.
		emit({
			event: "step_update",
			step_update: {
				step_index: 2,
				step_type: "agent_response",
				state: "DONE",
				text_delta: "TURN2_USAGE_ANSWER",
				usage: { input_tokens: 50, output_tokens: 15, cache_read_tokens: 12 },
			},
		});
		emit({
			event: "result",
			result: {
				conversation_id: cid,
				status: "SUCCESS",
				response: "TURN2_USAGE_ANSWER",
				num_turns: 2,
				usage: { input_tokens: 90 },
			},
		});
	}
}
`,
			"fake-agy-partial-usage",
		);

		let injected = false;
		const res = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Turn 1 partial usage test",
			index: 0,
			id: "PartialUsageAgy",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: false,
			settings: makeSettings(partialUsageBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!injected && p.currentTool === "bash") {
					injected = true;
					void IrcBus.global().send({
						from: "PeerUsage",
						to: "PartialUsageAgy",
						body: "Turn 2 partial usage follow-up",
					});
				}
			},
		});

		expect(res.exitCode).toBe(0);
		expect(res.output).toBe("TURN2_USAGE_ANSWER");
		expect(res.usage?.input).toBe(90);
		expect(res.usage?.output).toBe(25);
		expect(res.usage?.cacheRead).toBe(20);
		expect(res.contextTokens).toBe(62);

		const transcriptLines = (await fs.readFile(path.join(artifactsDir, "PartialUsageAgy.jsonl"), "utf8"))
			.trim()
			.split("\n")
			.map(
				l =>
					JSON.parse(l) as {
						type: string;
						message?: {
							role?: string;
							content?: Array<{ text?: string }>;
							usage?: { input?: number; output?: number; cacheRead?: number };
						};
					},
			);
		const assistantEntries = transcriptLines.filter(e => e.type === "message" && e.message?.role === "assistant");
		const turn1Entry = assistantEntries.find(e =>
			(e.message?.content ?? []).some(c => c.text?.includes("TURN1_USAGE_ANSWER")),
		);
		const turn2Entry = assistantEntries.find(e =>
			(e.message?.content ?? []).some(c => c.text?.includes("TURN2_USAGE_ANSWER")),
		);
		expect(turn1Entry?.message?.usage?.input).toBe(40);
		expect(turn1Entry?.message?.usage?.output).toBe(10);
		expect(turn1Entry?.message?.usage?.cacheRead).toBe(8);
		expect(turn2Entry?.message?.usage?.input).toBe(50);
		expect(turn2Entry?.message?.usage?.output).toBe(15);
		expect(turn2Entry?.message?.usage?.cacheRead).toBe(12);

		// Two-turn same-process run with result-only usage (no agent_response DONE frames): cumulative usage
		// and requests reconcile cleanly, while contextTokens must stay undefined instead of reporting Turn 1's 48.
		const resultOnlyTwoTurnBin = await createFakeAgyBinary(
			dir,
			`
import * as readline from "node:readline";
const cid = "conv-result-only-two-turns";
const emit = obj => process.stdout.write(JSON.stringify(obj) + "\\n");
emit({ event: "init", conversation_id: cid, init: { model: "gemini-4-neon-medium" } });

const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const linesIter = rl[Symbol.asyncIterator]();
const readPromptLine = async () => {
	while (true) {
		const next = await linesIter.next();
		if (next.done) return null;
		if (next.value.trim()) return JSON.parse(next.value)?.message?.content ?? "";
	}
};

const first = await readPromptLine();
if (first !== null) {
	emit({
		event: "step_update",
		step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "ACTIVE", tool_info: { name: "bash", parameters: { command: "wait" } } },
	});
	const second = await readPromptLine();
	emit({
		event: "step_update",
		step_update: { step_index: 1, step_type: "tool", tool_name: "bash", state: "DONE", tool_info: { name: "bash", output: "ok" } },
	});
	emit({
		event: "result",
		result: {
			conversation_id: cid,
			status: "SUCCESS",
			response: "RESULT_ONLY_TURN1",
			num_turns: 1,
			usage: { input_tokens: 40, output_tokens: 10, cache_read_tokens: 8 },
		},
	});
	if (second !== null) {
		emit({
			event: "result",
			result: {
				conversation_id: cid,
				status: "SUCCESS",
				response: "RESULT_ONLY_TURN2",
				num_turns: 2,
				usage: { input_tokens: 90, output_tokens: 25, cache_read_tokens: 20 },
			},
		});
	}
}
`,
			"fake-agy-result-only-two-turns",
		);

		let injectedTwoTurn = false;
		const twoTurnRes = await runSubprocess({
			cwd: dir,
			agent: makeAgent(),
			task: "Turn 1 result-only context check",
			index: 0,
			id: "ResultOnlyTwoTurnAgy",
			parentAgentId: MAIN_AGENT_ID,
			artifactsDir,
			sessionFile: parentSession,
			keepAlive: false,
			settings: makeSettings(resultOnlyTwoTurnBin, { task: "gemini-4-neon-medium" }),
			onProgress: p => {
				if (!injectedTwoTurn && p.currentTool === "bash") {
					injectedTwoTurn = true;
					void IrcBus.global().send({
						from: "PeerTwoTurn",
						to: "ResultOnlyTwoTurnAgy",
						body: "Turn 2 result-only follow-up",
					});
				}
			},
		});

		expect(twoTurnRes.exitCode).toBe(0);
		expect(twoTurnRes.output).toBe("RESULT_ONLY_TURN2");
		expect(twoTurnRes.requests).toBe(2);
		expect(twoTurnRes.usage?.input).toBe(90);
		expect(twoTurnRes.usage?.output).toBe(25);
		expect(twoTurnRes.usage?.cacheRead).toBe(20);
		expect(twoTurnRes.contextTokens).toBeUndefined();
	}, 30_000);
});
