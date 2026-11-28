import { Database } from "bun:sqlite";
import { describe, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as path from "node:path";
import {
	getDashboardStats,
	getProviderDashboardStats,
	getRequestDetails,
	getToolDashboardStats,
	syncAllSessions,
} from "@oh-my-pi/omp-stats/aggregator";
import { getRecentRequests, initDb } from "@oh-my-pi/omp-stats/db";
import { StatsLive } from "@oh-my-pi/omp-stats/live";
import type { LiveStatus } from "@oh-my-pi/omp-stats/shared-types";
import { getSessionsDir } from "@oh-my-pi/pi-utils";
import { installStatsTestIsolation } from "./helpers/temp-agent";

const AGY_CLI_API = "agy-cli";
const AGY_CLI_PROVIDER = "agy-cli";
const STATS_AGY_DATA_DIR_ENV = "OMP_STATS_AGY_DATA_DIR";

const isolation = installStatsTestIsolation("@pi-stats-agy-");

function agyDataDir(): string {
	const dir = isolation.current()?.join("agy-data");
	if (!dir) throw new Error("missing isolated temp dir");
	return dir;
}

async function createAgySummaryDb(
	rows: Array<{
		conversationId: string;
		workspaceUris: string;
		nestingDepth?: number;
		parentConversationId?: string;
	}>,
): Promise<string> {
	const root = agyDataDir();
	await fs.mkdir(root, { recursive: true });
	const dbPath = path.join(root, "conversation_summaries.db");
	const db = new Database(dbPath);
	try {
		db.run(`
			CREATE TABLE IF NOT EXISTS conversation_summaries (
				conversation_id TEXT PRIMARY KEY,
				workspace_uris TEXT NOT NULL DEFAULT '[]',
				nesting_depth INTEGER NOT NULL DEFAULT 0,
				parent_conversation_id TEXT NOT NULL DEFAULT ''
			)
		`);
		const insert = db.query(`
			INSERT OR REPLACE INTO conversation_summaries (conversation_id, workspace_uris, nesting_depth, parent_conversation_id)
			VALUES (?, ?, ?, ?)
		`);
		for (const row of rows) {
			insert.run(row.conversationId, row.workspaceUris, row.nestingDepth ?? 0, row.parentConversationId ?? "");
		}
	} finally {
		db.close();
	}
	return dbPath;
}

async function writeAgyTranscript(conversationId: string, entries: Array<Record<string, unknown>>): Promise<string> {
	const transcriptPath = path.join(
		agyDataDir(),
		"brain",
		conversationId,
		".system_generated",
		"logs",
		"transcript.jsonl",
	);
	await fs.mkdir(path.dirname(transcriptPath), { recursive: true });
	await Bun.write(transcriptPath, `${entries.map(e => JSON.stringify(e)).join("\n")}\n`);
	return transcriptPath;
}

function nextSync(live: StatsLive): Promise<LiveStatus> {
	const before = live.status().sync.lastSyncedAt;
	const { promise, resolve } = Promise.withResolvers<LiveStatus>();
	const unsubscribe = live.subscribe(status => {
		if (status.sync.phase === "idle" && status.sync.lastSyncedAt !== before) {
			unsubscribe();
			resolve(status);
		}
	});
	return promise;
}

describe("agy transcript ingestion", () => {
	it("ingests standalone agy sessions, nested agy subagents, and OMP-delegated subagents with folder and unpriced cost attribution", async () => {
		const now = new Date();
		const ts1 = new Date(now.getTime() - 60_000).toISOString();
		const ts2 = new Date(now.getTime() - 50_000).toISOString();
		const ts3 = new Date(now.getTime() - 40_000).toISOString();
		const ts4 = new Date(now.getTime() - 30_000).toISOString();

		await createAgySummaryDb([
			{ conversationId: "conv-standalone", workspaceUris: JSON.stringify(["file:///root/my-app"]) },
			{
				conversationId: "conv-nested",
				workspaceUris: JSON.stringify(["file:///root/my-app"]),
				nestingDepth: 1,
				parentConversationId: "conv-standalone",
			},
			{ conversationId: "conv-omp-delegated", workspaceUris: JSON.stringify(["file:///root/efm-enyo-intern"]) },
			{ conversationId: "conv-omp-spilled", workspaceUris: JSON.stringify(["file:///root/nuli"]) },
		]);

		await writeAgyTranscript("conv-standalone", [
			{
				step_index: 0,
				source: "USER",
				type: "USER_INPUT",
				status: "DONE",
				created_at: ts1,
				content: "<USER_REQUEST>\nCheck the server status please\n</USER_REQUEST>",
			},
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts2,
				model: "gemini-4-neon-high",
				input_tokens: 1000,
				cache_read_tokens: 5000,
				output_tokens: 200,
				content: "Checking server status now.",
				tool_calls: [{ name: "run_command", args: { CommandLine: "uptime" } }],
			},
			{
				step_index: 2,
				source: "SYSTEM",
				type: "CHECKPOINT",
				status: "DONE",
				created_at: ts2,
				input_tokens: 9999,
				output_tokens: 999,
				content: "# Resuming from a compaction",
			},
		]);

		await writeAgyTranscript("conv-nested", [
			{
				step_index: 0,
				source: "USER",
				type: "USER_INPUT",
				status: "DONE",
				created_at: ts2,
				content: "Research the codebase",
			},
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts3,
				model: "gemini-4-neon-medium",
				input_tokens: 400,
				cache_read_tokens: 100,
				output_tokens: 50,
			},
		]);

		await writeAgyTranscript("conv-omp-delegated", [
			{
				step_index: 0,
				source: "USER",
				type: "USER_INPUT",
				status: "DONE",
				created_at: ts3,
				content:
					'<USER_REQUEST>\nYou are executing a delegated "task" subagent task inside a larger workflow.\n</USER_REQUEST>',
			},
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts4,
				model: "gemini-4-neon-medium",
				input_tokens: 800,
				cache_read_tokens: 2000,
				output_tokens: 150,
				tool_calls: [
					{ name: "read_file", args: { Path: "/root/efm-enyo-intern/README.md" } },
					{ name: "run_command", args: { CommandLine: "bun test" } },
				],
			},
		]);

		await writeAgyTranscript("conv-omp-spilled", [
			{
				step_index: 0,
				source: "USER",
				type: "USER_INPUT",
				status: "DONE",
				created_at: ts3,
				content:
					"Read the complete task instructions from /tmp/Worker.agy-prompt-0-12345.txt and execute them thoroughly.",
			},
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts4,
				model: "gemini-4-neon-low",
				input_tokens: 300,
				cache_read_tokens: 0,
				output_tokens: 60,
			},
		]);

		await syncAllSessions({ workers: 1 });

		const dashboard = await getDashboardStats("24h");
		expect(dashboard.overall.totalRequests).toBe(4);
		expect(dashboard.overall.totalInputTokens).toBe(2500);
		expect(dashboard.overall.totalOutputTokens).toBe(460);
		expect(dashboard.overall.totalCacheReadTokens).toBe(7100);
		// Priced from bundled Google/OpenRouter Gemini 4 Neon peer card ($3/M in, $15/M out, $0.15/M cacheRead).
		expect(dashboard.overall.totalCost).toBeCloseTo(2500 * 3e-6 + 460 * 15e-6 + 7100 * 0.15e-6, 9);
		expect(dashboard.overall.unpricedRequests).toBe(0);

		const byAgentType = new Map(dashboard.byAgentType.map(a => [a.agentType, a]));
		expect(byAgentType.get("main")?.totalRequests).toBe(1);
		expect(byAgentType.get("subagent")?.totalRequests).toBe(3);

		const folders = new Map(dashboard.byFolder.map(f => [f.folder, f]));
		expect(folders.get("/root/my-app")?.totalRequests).toBe(2);
		expect(folders.get("/root/efm-enyo-intern")?.totalRequests).toBe(1);
		expect(folders.get("/root/nuli")?.totalRequests).toBe(1);

		const providers = await getProviderDashboardStats("24h");
		expect(providers.providers.map(p => p.provider)).toEqual([AGY_CLI_PROVIDER]);

		const tools = await getToolDashboardStats("24h");
		const toolCounts = new Map(tools.byTool.map(t => [t.tool, t.calls]));
		expect(toolCounts.get("run_command")).toBe(2);
		expect(toolCounts.get("read_file")).toBe(1);

		// Request details modal resolves the synthetic agy entry back to the PLANNER_RESPONSE payload.
		const recent = getRecentRequests(10);
		const standaloneReq = recent.find(r => r.model === "gemini-4-neon-high");
		expect(standaloneReq?.api).toBe(AGY_CLI_API);
		expect(standaloneReq?.costUnpriced).toBe(false);
		expect(standaloneReq?.usage.cost.total).toBeCloseTo(1000 * 3e-6 + 200 * 15e-6 + 5000 * 0.15e-6, 9);
		const details = await getRequestDetails(standaloneReq!.id!);
		expect(details).not.toBeNull();
		expect(JSON.stringify(details?.output)).toContain("Checking server status now.");
	});

	it("never double-counts OMP child agy-cli mirror transcripts when brain ingestion is active, and falls back when disabled", async () => {
		const nowIso = new Date().toISOString();
		const sessionDir = path.join(getSessionsDir(), "--root--efm-enyo-intern--", "1700000000000_parent");
		await fs.mkdir(sessionDir, { recursive: true });
		const childPath = path.join(sessionDir, "WorkerAgy.jsonl");
		const childEntries = [
			{ type: "session", version: 3, id: "child-1", timestamp: nowIso, cwd: "/root/efm-enyo-intern" },
			{
				type: "model_change",
				id: "agy-1",
				parentId: null,
				timestamp: nowIso,
				model: "agy-cli/gemini-4-neon-medium",
			},
			{
				type: "message",
				id: "agy-2",
				parentId: "agy-1",
				timestamp: nowIso,
				message: {
					role: "assistant",
					content: [{ type: "toolCall", id: "agy-call-0-1", name: "read_file", arguments: { Path: "a.ts" } }],
					api: AGY_CLI_API,
					provider: AGY_CLI_PROVIDER,
					model: "gemini-4-neon-medium",
					stopReason: "toolUse",
					timestamp: Date.now(),
					usage: {
						input: 500,
						output: 40,
						cacheRead: 1000,
						cacheWrite: 0,
						totalTokens: 1540,
						cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
					},
				},
			},
		];
		await Bun.write(childPath, `${childEntries.map(e => JSON.stringify(e)).join("\n")}\n`);

		await createAgySummaryDb([
			{ conversationId: "conv-dedupe", workspaceUris: JSON.stringify(["file:///root/efm-enyo-intern"]) },
		]);
		await writeAgyTranscript("conv-dedupe", [
			{
				step_index: 0,
				source: "USER",
				type: "USER_INPUT",
				status: "DONE",
				created_at: nowIso,
				content: 'You are executing a delegated "task" subagent task inside a larger workflow.',
			},
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: nowIso,
				model: "gemini-4-neon-medium",
				input_tokens: 500,
				cache_read_tokens: 1000,
				output_tokens: 40,
				tool_calls: [{ name: "read_file", args: { Path: "a.ts" } }],
			},
		]);

		// With brain ingestion active (default), only the canonical brain/ transcript is counted — zero duplication!
		await syncAllSessions({ workers: 1 });
		const activeDashboard = await getDashboardStats("24h");
		expect(activeDashboard.overall.totalRequests).toBe(1);
		expect(activeDashboard.overall.totalInputTokens).toBe(500);
		const activeTools = await getToolDashboardStats("24h");
		expect(activeTools.byTool.find(t => t.tool === "read_file")?.calls).toBe(1);

		// When brain ingestion is explicitly disabled (`OMP_STATS_AGY_DATA_DIR=""`), the OMP child mirror is ingested cleanly.
		const prev = process.env[STATS_AGY_DATA_DIR_ENV];
		process.env[STATS_AGY_DATA_DIR_ENV] = "";
		try {
			await fs.rm(path.join(agyDataDir(), "brain"), { recursive: true, force: true });
			// Force fresh parse of childPath on a clean file name to verify fallback ingestion path directly.
			const fallbackChild = path.join(sessionDir, "WorkerFallback.jsonl");
			await Bun.write(fallbackChild, `${childEntries.map(e => JSON.stringify(e)).join("\n")}\n`);
			await syncAllSessions({ files: [fallbackChild], workers: 1 });
			const recent = getRecentRequests(10);
			expect(recent.some(r => r.sessionFile === fallbackChild && r.provider === AGY_CLI_PROVIDER)).toBe(true);
		} finally {
			if (prev === undefined) delete process.env[STATS_AGY_DATA_DIR_ENV];
			else process.env[STATS_AGY_DATA_DIR_ENV] = prev;
		}
	});

	it("tails appended agy turns incrementally without consuming partial lines and rebuilds folder when summary DB appears", async () => {
		const ts1 = new Date(Date.now() - 20_000).toISOString();
		const ts2 = new Date(Date.now() - 10_000).toISOString();
		const transcriptPath = await writeAgyTranscript("conv-inc", [
			{ step_index: 0, source: "USER", type: "USER_INPUT", status: "DONE", created_at: ts1, content: "hello" },
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts1,
				model: "gemini-4-neon-high",
				input_tokens: 100,
				cache_read_tokens: 0,
				output_tokens: 25,
			},
		]);

		// Append an incomplete JSON line (no trailing newline) simulating a live write in progress.
		const secondTurn = JSON.stringify({
			step_index: 2,
			source: "MODEL",
			type: "PLANNER_RESPONSE",
			status: "DONE",
			created_at: ts2,
			model: "gemini-4-neon-high",
			input_tokens: 200,
			cache_read_tokens: 50,
			output_tokens: 40,
		});
		const half = Math.floor(secondTurn.length / 2);
		await fs.appendFile(transcriptPath, secondTurn.slice(0, half));

		// First sync (before conversation_summaries.db exists) records turn 1 with fallback folder "agy".
		await syncAllSessions({ workers: 1 });
		let dash = await getDashboardStats("24h");
		expect(dash.overall.totalRequests).toBe(1);
		expect(dash.byFolder.map(f => f.folder)).toEqual(["agy"]);

		// Complete the second line AND populate conversation_summaries.db -> rebuilds folder and adds turn 2.
		await fs.appendFile(transcriptPath, `${secondTurn.slice(half)}\n`);
		await createAgySummaryDb([
			{ conversationId: "conv-inc", workspaceUris: JSON.stringify(["file:///root/resolved-project"]) },
		]);

		await syncAllSessions({ workers: 1 });
		dash = await getDashboardStats("24h");
		expect(dash.overall.totalRequests).toBe(2);
		expect(dash.overall.totalInputTokens).toBe(300);
		expect(dash.byFolder.map(f => f.folder)).toEqual(["/root/resolved-project"]);
	});

	it("watches agy brain directory in StatsLive and syncs newly written transcripts", async () => {
		await fs.mkdir(path.join(agyDataDir(), "brain"), { recursive: true });
		await createAgySummaryDb([
			{ conversationId: "conv-live", workspaceUris: JSON.stringify(["file:///root/live-proj"]) },
		]);

		const live = new StatsLive();
		try {
			const initial = nextSync(live);
			live.start();
			await initial;

			const targeted = nextSync(live);
			await writeAgyTranscript("conv-live", [
				{
					step_index: 1,
					source: "MODEL",
					type: "PLANNER_RESPONSE",
					status: "DONE",
					created_at: new Date().toISOString(),
					model: "gemini-4-neon-high",
					input_tokens: 777,
					cache_read_tokens: 0,
					output_tokens: 33,
				},
			]);
			await targeted;

			await initDb();
			const recent = getRecentRequests(10);
			expect(recent.map(r => [r.provider, r.model, r.folder, r.usage.input])).toEqual([
				[AGY_CLI_PROVIDER, "gemini-4-neon-high", "/root/live-proj", 777],
			]);
		} finally {
			live.stop();
		}
	}, 20_000);

	it("does not leak host ~/.gemini/antigravity-cli transcripts into isolated OMP profiles when agy env overrides are unset", async () => {
		const prevStatsAgy = process.env[STATS_AGY_DATA_DIR_ENV];
		const prevAgyApp = process.env.AGY_APP_DATA_DIR;
		delete process.env[STATS_AGY_DATA_DIR_ENV];
		delete process.env.AGY_APP_DATA_DIR;
		try {
			const { getAgyAppDataDir } = await import("@oh-my-pi/omp-stats/agy-parser");
			// Isolated test profile (via installStatsTestIsolation) must not auto-discover host ~/.gemini/antigravity-cli.
			expect(getAgyAppDataDir()).toBeNull();

			// Syncing an isolated profile with one OMP fixture must count only that fixture.
			const nowIso = new Date().toISOString();
			const fixturePath = path.join(getSessionsDir(), "--tmp--", "isolated-smoke.jsonl");
			await fs.mkdir(path.dirname(fixturePath), { recursive: true });
			await Bun.write(
				fixturePath,
				`${JSON.stringify({ type: "session", version: 3, id: "iso-1", timestamp: nowIso, cwd: "/tmp" })}\n${JSON.stringify(
					{
						type: "model_usage",
						id: "iso-m1",
						parentId: null,
						timestamp: nowIso,
						purpose: "title",
						api: "openai-responses",
						provider: "openai",
						model: "gpt-5",
						stopReason: "stop",
						usage: {
							input: 1,
							output: 1,
							cacheRead: 0,
							cacheWrite: 0,
							totalTokens: 2,
							cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
						},
					},
				)}\n`,
			);
			await syncAllSessions({ workers: 1 });
			const dash = await getDashboardStats("24h");
			expect(dash.overall.totalRequests).toBe(1);
		} finally {
			if (prevStatsAgy === undefined) delete process.env[STATS_AGY_DATA_DIR_ENV];
			else process.env[STATS_AGY_DATA_DIR_ENV] = prevStatsAgy;
			if (prevAgyApp === undefined) delete process.env.AGY_APP_DATA_DIR;
			else process.env.AGY_APP_DATA_DIR = prevAgyApp;
		}
	});

	it("prices known agy-cli peer models via bundled catalog, keeps unknown models unpriced, and backfills existing zero-cost rows", async () => {
		const ts = new Date(Date.now() - 5_000).toISOString();
		await writeAgyTranscript("conv-pricing", [
			{
				step_index: 1,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts,
				model: "gemini-4-neon-high",
				input_tokens: 1_000_000,
				cache_read_tokens: 1_000_000,
				output_tokens: 100_000,
			},
			{
				step_index: 2,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts,
				model: "claude-opus-5-5-medium",
				input_tokens: 500_000,
				cache_read_tokens: 0,
				output_tokens: 50_000,
			},
			{
				step_index: 3,
				source: "MODEL",
				type: "PLANNER_RESPONSE",
				status: "DONE",
				created_at: ts,
				model: "unknown-future-model-high",
				input_tokens: 10_000,
				cache_read_tokens: 0,
				output_tokens: 1_000,
			},
		]);

		await syncAllSessions({ workers: 1 });
		const dash = await getDashboardStats("24h");
		const byModel = new Map(dash.byModel.map(m => [m.model, m]));

		// Gemini 4 Neon: $3/M in + $0.15/M cacheRead + $15/M out => $3 + $0.15 + $1.50 = $4.65
		// Prompt savings ratio: ($6.00 uncached - $3.15 cached) / $6.00 = 0.475 (47.5%)
		expect(byModel.get("gemini-4-neon-high")?.totalCost).toBeCloseTo(4.65, 6);
		expect(byModel.get("gemini-4-neon-high")?.unpricedRequests).toBe(0);
		expect(byModel.get("gemini-4-neon-high")?.cacheSavings).toBeCloseTo(0.475, 6);

		// Claude Opus 5.5: $4/M in * 0.5 + $20/M out * 0.05 => $2 + $1 = $3.00
		expect(byModel.get("claude-opus-5-5-medium")?.totalCost).toBeCloseTo(3.0, 6);
		expect(byModel.get("claude-opus-5-5-medium")?.unpricedRequests).toBe(0);

		// Unknown model stays unpriced subscription usage rather than $0 free.
		expect(byModel.get("unknown-future-model-high")?.totalCost).toBe(0);
		expect(byModel.get("unknown-future-model-high")?.unpricedRequests).toBe(1);

		// Simulate pre-PR32 DB rows (zero cost, cost_unpriced=1, backfill key absent) and verify initDb backfills them.
		const { closeDb } = await import("@oh-my-pi/omp-stats/db");
		const { getStatsDbPath } = await import("@oh-my-pi/pi-utils");
		closeDb();
		const raw = new Database(getStatsDbPath());
		raw.run(
			"UPDATE messages SET cost_input = 0, cost_output = 0, cost_cache_read = 0, cost_cache_write = 0, cost_total = 0, cost_no_cache_input = 0, cost_unpriced = 1 WHERE provider = 'agy-cli'",
		);
		raw.run("DELETE FROM meta WHERE key = 'agy_cli_catalog_cost_v1'");
		raw.close();

		await initDb();
		const afterBackfill = await getDashboardStats("24h");
		const backfilledNeon = afterBackfill.byModel.find(m => m.model === "gemini-4-neon-high");
		expect(backfilledNeon?.totalCost).toBeCloseTo(4.65, 6);
		expect(backfilledNeon?.unpricedRequests).toBe(0);
		expect(backfilledNeon?.cacheSavings).toBeCloseTo(0.475, 6);
	});
});
