/**
 * Contract: reading a topic's conversation (`read history://<id>`) pages it:
 * one read returns at most the read tool's byte budget, however large the
 * topic, with a continuation selector for the next page, so no prompt pulls
 * a whole long topic into context in one read.
 */
import { afterEach, expect, it } from "bun:test";
import * as path from "node:path";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { InternalUrlRouter } from "@oh-my-pi/pi-coding-agent/internal-urls";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { ReadTool } from "@oh-my-pi/pi-coding-agent/tools/read";
import { formatOutputNotice } from "@oh-my-pi/pi-tui/tools/output-meta";
import { DEFAULT_MAX_BYTES } from "@oh-my-pi/pi-tui/tools/streaming-output";
import { TempDir } from "@oh-my-pi/pi-utils";

const temp = TempDir.createSync("@omp-topic-history-");
/** One page: the byte budget plus the hashline header and notice. */
const PAGE_MAX = DEFAULT_MAX_BYTES + 4096;

afterEach(() => {
	AgentRegistry.resetGlobalForTests();
	InternalUrlRouter.resetForTests();
});

it("pages a huge topic conversation at the read byte budget, with a continuation", async () => {
	const manager = SessionManager.create(temp.path(), path.join(temp.path(), "topics"));
	const paragraph = "The owner asked about clause 4 again and the topic answered at length. ".repeat(20);
	for (let turn = 0; turn < 400; turn++) {
		manager.appendMessage({ role: "user", content: `question ${turn}: ${paragraph}`, timestamp: turn });
	}
	await manager.ensureOnDisk();
	await manager.flush();
	AgentRegistry.global().register({
		id: "Diokan",
		displayName: "Diokan",
		kind: "sub",
		parentId: "Main",
		session: null,
		sessionFile: manager.getSessionFile()!,
		status: "idle",
	});
	const whole = await InternalUrlRouter.instance().resolve("history://Diokan");
	expect(Buffer.byteLength(String(whole.content))).toBeGreaterThan(10 * DEFAULT_MAX_BYTES);

	const session = {
		cwd: temp.path(),
		hasUI: false,
		getSessionFile: () => null,
		getSessionSpawns: () => "*",
		getArtifactsDir: () => null,
		settings: Settings.isolated(),
	} as unknown as ToolSession;
	const read = async (target: string) => {
		const result = await new ReadTool(session).execute("read-history", { path: target });
		const text = result.content.flatMap(block => (block.type === "text" ? [block.text] : [])).join("\n");
		return { text, notice: formatOutputNotice(result.details?.meta) };
	};
	const first = await read("history://Diokan");
	expect(Buffer.byteLength(first.text)).toBeLessThan(PAGE_MAX);
	expect(first.text).toContain("question 0:");
	const next = first.notice.match(/Use :(\d+)- to continue/)?.[1];
	expect(next).toBeDefined();
	// An explicit huge range is capped the same way.
	expect(Buffer.byteLength((await read("history://Diokan:1-1000000")).text)).toBeLessThan(PAGE_MAX);
	const second = await read(`history://Diokan:${next}-`);
	expect(Buffer.byteLength(second.text)).toBeLessThan(PAGE_MAX);
	expect(second.text).not.toContain("question 0:");
	expect(second.text).toContain("question ");
});
