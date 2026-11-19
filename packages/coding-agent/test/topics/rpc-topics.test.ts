/**
 * Contract: RPC mode treats exactly the seven `topics_*` commands as topic
 * commands; a mistyped one takes the ordinary unknown-command path instead of
 * reporting success. Drives a real SDK session in `runRpcMode` over stdio
 * (`fixtures/progress-events-rpc-agent.ts`, topics disabled).
 */
import { afterEach, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

const FIXTURE = path.join(import.meta.dir, "..", "fixtures", "progress-events-rpc-agent.ts");

let child: Bun.Subprocess<"pipe", "pipe", "ignore"> | undefined;
let directory: string | undefined;

afterEach(async () => {
	child?.kill();
	await child?.exited;
	child = undefined;
	if (directory) await removeWithRetries(directory);
	directory = undefined;
});

/** Send `frames` as JSON lines and collect the responses to their ids. */
async function exchange(frames: Array<{ id: string; type: string } & Record<string, unknown>>) {
	directory = await fs.mkdtemp(path.join(os.tmpdir(), "omp-rpc-topics-"));
	child = Bun.spawn([process.execPath, FIXTURE], {
		cwd: directory,
		env: { ...process.env, PI_CODING_AGENT_DIR: directory, PI_NO_TITLE: "1" },
		stdin: "pipe",
		stdout: "pipe",
		stderr: "ignore",
	});
	child.stdin.write(frames.map(frame => `${JSON.stringify(frame)}\n`).join(""));
	child.stdin.flush();
	const responses = new Map<string, Record<string, unknown>>();
	const decoder = new TextDecoder();
	let buffer = "";
	for await (const chunk of child.stdout) {
		buffer += decoder.decode(chunk, { stream: true });
		let newline = buffer.indexOf("\n");
		while (newline >= 0) {
			const frame = JSON.parse(buffer.slice(0, newline)) as Record<string, unknown>;
			buffer = buffer.slice(newline + 1);
			newline = buffer.indexOf("\n");
			if (frame.type === "response" && typeof frame.id === "string") responses.set(frame.id, frame);
		}
		if (frames.every(frame => responses.has(frame.id))) break;
	}
	return responses;
}

it("answers a mistyped topics command as unknown, and a real one through the topics path", async () => {
	const responses = await exchange([
		{ id: "typo", type: "topics_clsoe", topicId: "Diokan" },
		{ id: "real", type: "topics_close", topicId: "Diokan" },
	]);
	expect(responses.get("typo")).toMatchObject({
		command: "topics_clsoe",
		success: false,
		error: "Unknown command: topics_clsoe",
	});
	expect(responses.get("real")).toMatchObject({
		command: "topics_close",
		success: false,
		error: "Topics are disabled; enable the topics.enabled setting.",
	});
}, 60_000);
