/**
 * Contract: the topics control socket is an owner-only door that a host
 * opens with its per-launch token and that reaches topics, nothing else.
 */
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import type { RpcSubagentMessagesResult } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-types";
import { startTopicsControl, TOPICS_CONTROL_PROTOCOL } from "@oh-my-pi/pi-coding-agent/topics/control";
import type { TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import type { WorkerChange } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import { TempDir } from "@oh-my-pi/pi-utils";

const TOKEN = "s3cret-token";

function fakeHost(sent: Array<{ id: string; message: string }>): TopicsHost {
	const listeners = new Set<(change: WorkerChange) => void>();
	return {
		list: () => [
			{ id: "Diokan", title: "Diokan", state: "idle", turns: 1, queued: 0, createdAt: 1, lastActivityAt: 2 },
		],
		create: async () => ({ id: "New" }),
		async send(id, message) {
			sent.push({ id, message });
			for (const listener of listeners)
				listener({ id, ownerId: "Main", parentSessionId: "p", reason: "turn-started" });
			return { id, mode: "turn" };
		},
		read: async () => ({}) as RpcSubagentMessagesResult,
		close: async () => {},
		reopen: async () => {},
		rename: async () => {},
		restore: async () => 0,
		onChange(listener) {
			listeners.add(listener);
			return () => listeners.delete(listener);
		},
	};
}

/** Minimal line client: send frames, collect parsed reply frames. */
async function connect(socketPath: string): Promise<{
	send: (frame: object) => void;
	next: () => Promise<Record<string, unknown>>;
	closed: Promise<void>;
	end: () => void;
}> {
	const frames: Record<string, unknown>[] = [];
	const waiters: Array<(frame: Record<string, unknown>) => void> = [];
	const closed = Promise.withResolvers<void>();
	let buffer = "";
	const socket = await Bun.connect({
		unix: socketPath,
		socket: {
			data(_socket, chunk) {
				buffer += chunk.toString("utf8");
				let newline = buffer.indexOf("\n");
				while (newline >= 0) {
					const frame = JSON.parse(buffer.slice(0, newline)) as Record<string, unknown>;
					buffer = buffer.slice(newline + 1);
					newline = buffer.indexOf("\n");
					const waiter = waiters.shift();
					if (waiter) waiter(frame);
					else frames.push(frame);
				}
			},
			close() {
				closed.resolve();
			},
		},
	});
	return {
		send: frame => socket.write(`${JSON.stringify(frame)}\n`),
		next: () => {
			const queued = frames.shift();
			if (queued) return Promise.resolve(queued);
			const { promise, resolve } = Promise.withResolvers<Record<string, unknown>>();
			waiters.push(resolve);
			return promise;
		},
		closed: closed.promise,
		end: () => socket.end(),
	};
}

describe("topics control socket", () => {
	let tempDir: TempDir;
	let socketPath: string;
	let stop: (() => void) | undefined;
	let sent: Array<{ id: string; message: string }>;

	beforeEach(() => {
		tempDir = TempDir.createSync("@omp-topics-control-");
		socketPath = path.join(tempDir.path(), "control.sock");
		sent = [];
		const host = fakeHost(sent);
		stop = startTopicsControl({ path: socketPath, token: TOKEN, topics: async () => host });
	});

	afterEach(() => {
		stop?.();
		tempDir.removeSync();
	});

	it("creates the socket readable and writable by its owner only", () => {
		expect(fs.statSync(socketPath).mode & 0o777).toBe(0o600);
	});

	it("refuses a client with the wrong token and closes the connection", async () => {
		const client = await connect(socketPath);
		client.send({ type: "hello", token: "wrong" });
		expect(await client.next()).toEqual({ type: "error", error: "Unauthorized." });
		await client.closed;
		expect(sent).toEqual([]);
	});

	it("refuses commands sent before the hello", async () => {
		const client = await connect(socketPath);
		client.send({ id: "1", type: "topics_send", topicId: "Diokan", message: "hi" });
		expect(await client.next()).toEqual({ type: "error", error: "Unauthorized." });
		await client.closed;
		expect(sent).toEqual([]);
	});

	it("serves topic commands and change events after the hello, and nothing else", async () => {
		const client = await connect(socketPath);
		client.send({ type: "hello", token: TOKEN });
		expect(await client.next()).toMatchObject({ type: "hello", protocol: TOPICS_CONTROL_PROTOCOL });

		client.send({ id: "1", type: "topics_list" });
		expect(await client.next()).toMatchObject({ id: "1", success: true, data: { topics: [{ id: "Diokan" }] } });

		client.send({ id: "2", type: "subscribe" });
		expect(await client.next()).toMatchObject({ id: "2", success: true });
		client.send({ id: "3", type: "topics_send", topicId: "Diokan", message: "status?" });
		const replies = [await client.next(), await client.next()];
		expect(replies).toContainEqual(expect.objectContaining({ type: "topic_update" }));
		expect(replies).toContainEqual(
			expect.objectContaining({ id: "3", success: true, data: { topicId: "Diokan", mode: "turn" } }),
		);
		expect(sent).toEqual([{ id: "Diokan", message: "status?" }]);

		client.send({ id: "4", type: "prompt", message: "do something in the main session" });
		expect(await client.next()).toMatchObject({ id: "4", success: false, error: "Unknown command." });
		client.end();
	});

	it("never replaces an existing socket path", () => {
		expect(() => startTopicsControl({ path: socketPath, token: TOKEN, topics: async () => undefined })).toThrow(
			"already exists",
		);
	});
});
