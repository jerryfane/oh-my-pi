/**
 * Contract: RPC mode treats exactly the `topics_*` commands as topic
 * commands; a mistyped one takes the ordinary unknown-command path instead of
 * reporting success. Refusals with a machine-readable reason carry `code`
 * next to the unchanged `error` text. Drives a real SDK session in
 * `runRpcMode` over stdio (`fixtures/progress-events-rpc-agent.ts`, topics
 * disabled; `fixtures/topics-rpc-agent.ts`, topics enabled).
 */
import { afterEach, expect, it } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { removeWithRetries } from "@oh-my-pi/pi-utils";

const DISABLED_FIXTURE = path.join(import.meta.dir, "..", "fixtures", "progress-events-rpc-agent.ts");
const TOPICS_FIXTURE = path.join(import.meta.dir, "..", "fixtures", "topics-rpc-agent.ts");

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
async function exchange(
	frames: Array<{ id: string; type: string } & Record<string, unknown>>,
	fixture: string = DISABLED_FIXTURE,
) {
	directory = await fs.mkdtemp(path.join(os.tmpdir(), "omp-rpc-topics-"));
	child = Bun.spawn([process.execPath, fixture], {
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

it("answers topics commands with topics disabled with code topics_disabled", async () => {
	const responses = await exchange([
		{ id: "list", type: "topics_list" },
		{ id: "organise", type: "topics_organise", by: "owner" },
	]);
	for (const id of ["list", "organise"]) {
		expect(responses.get(id)).toMatchObject({
			success: false,
			error: "Topics are disabled; enable the topics.enabled setting.",
			code: "topics_disabled",
		});
	}
}, 60_000);

it("carries code on suggestion, seed and organise refusals, and lists suggestions", async () => {
	const responses = await exchange(
		[
			{ id: "list", type: "topics_list" },
			{ id: "accept", type: "topics_accept_suggestion", suggestionId: "sg_unknown", by: "owner" },
			{ id: "dismiss", type: "topics_dismiss_suggestion", suggestionId: "sg_unknown", by: "owner" },
			{ id: "seed", type: "topics_new", title: "T", message: "m", seedMessageIds: ["no-such-entry"] },
			{ id: "organise", type: "topics_organise", by: "owner" },
			{ id: "organise-again", type: "topics_organise", by: "owner" },
		],
		TOPICS_FIXTURE,
	);
	expect(responses.get("list")).toMatchObject({ success: true, data: { topics: [], suggestions: [] } });
	expect(responses.get("accept")).toMatchObject({
		success: false,
		error: "Unknown suggestion: sg_unknown",
		code: "suggestion_not_found",
	});
	expect(responses.get("dismiss")).toMatchObject({ success: false, code: "suggestion_not_found" });
	expect(responses.get("seed")).toMatchObject({ success: false, code: "invalid_message_ids" });
	expect(responses.get("organise")).toMatchObject({ success: true, data: { queued: false } });
	expect(responses.get("organise-again")).toMatchObject({
		success: false,
		error: "An Organise turn is already queued or running.",
		code: "organise_running",
	});
}, 60_000);
