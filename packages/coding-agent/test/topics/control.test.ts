/**
 * Contract: the topics control socket is an owner-only door that a host
 * opens with its per-launch token and that reaches topics, nothing else.
 */
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import type { RpcSubagentMessagesResult } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-types";
import {
	MIN_TOKEN_LENGTH,
	startTopicsControl,
	TOPICS_CONTROL_PROTOCOL,
	takeTopicsControlRequest,
} from "@oh-my-pi/pi-coding-agent/topics/control";
import type { TopicsHost } from "@oh-my-pi/pi-coding-agent/topics/host";
import type { WorkerChange } from "@oh-my-pi/pi-coding-agent/vibe/runtime";
import { TempDir } from "@oh-my-pi/pi-utils";

const TOKEN = "s3cret-token-0123456789abcdefghijklmnop";
/** A transcript page larger than one socket write, to prove replies are not cut off. */
const BIG_TEXT = "x".repeat(3 * 1024 * 1024);

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
		read: async () =>
			({ messages: [{ role: "assistant", content: BIG_TEXT }] }) as unknown as RpcSubagentMessagesResult,
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

	beforeEach(async () => {
		tempDir = TempDir.createSync("@omp-topics-control-");
		socketPath = path.join(tempDir.path(), "control.sock");
		sent = [];
		const host = fakeHost(sent);
		stop = await startTopicsControl({ path: socketPath, token: TOKEN, topics: async () => host, helloTimeoutMs: 50 });
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

	it("delivers a reply larger than one socket write whole", async () => {
		const client = await connect(socketPath);
		client.send({ type: "hello", token: TOKEN });
		await client.next();
		client.send({ id: "big", type: "topics_read", topicId: "Diokan" });
		const reply = await client.next();
		expect(reply).toMatchObject({ id: "big", success: true });
		expect(JSON.stringify(reply)).toContain(BIG_TEXT);
		client.end();
	});

	it("closes a client that never says hello", async () => {
		const client = await connect(socketPath);
		await client.closed;
		expect(sent).toEqual([]);
	});

	it("refuses to take over a socket another server is still answering on", async () => {
		await expect(
			startTopicsControl({ path: socketPath, token: TOKEN, topics: async () => undefined }),
		).rejects.toThrow("in use");
	});

	it("replaces a dead socket left by a crashed run", async () => {
		stop?.();
		stop = undefined;
		// A SIGKILL leaves the socket inode behind with nobody listening. Bun unlinks
		// on stop(), so keep a hard link to the inode and put it back afterwards.
		const crashed = Bun.listen({ unix: socketPath, socket: { data() {} } });
		fs.linkSync(socketPath, `${socketPath}.keep`);
		crashed.stop(true);
		fs.renameSync(`${socketPath}.keep`, socketPath);
		expect(fs.lstatSync(socketPath).isSocket()).toBe(true);
		stop = await startTopicsControl({ path: socketPath, token: TOKEN, topics: async () => undefined });
		const client = await connect(socketPath);
		client.send({ type: "hello", token: TOKEN });
		expect(await client.next()).toMatchObject({ type: "hello" });
		client.end();
	});

	it("refuses a token shorter than the minimum", async () => {
		await expect(
			startTopicsControl({
				path: `${socketPath}.2`,
				token: "a".repeat(MIN_TOKEN_LENGTH - 1),
				topics: async () => undefined,
			}),
		).rejects.toThrow("at least");
	});
});

describe("topics control request from the environment", () => {
	it("reads the secret once and leaves nothing a child process could read it from", async () => {
		const tempDir = TempDir.createSync("@omp-topics-env-");
		try {
			const tokenFile = path.join(tempDir.path(), "token");
			fs.writeFileSync(tokenFile, `${TOKEN}\n`, { mode: 0o600 });
			const env: NodeJS.ProcessEnv = {
				HERDR_OMP_CONTROL: path.join(tempDir.path(), "c.sock"),
				HERDR_OMP_CONTROL_TOKEN_FILE: tokenFile,
			};

			const request = takeTopicsControlRequest(env);

			expect(request).toEqual({ path: path.join(tempDir.path(), "c.sock"), token: TOKEN });
			// A child running as the same user would read the file next; it must be gone.
			const child = Bun.spawnSync(["sh", "-c", `cat ${tokenFile} 2>/dev/null; echo end`]);
			expect(child.stdout.toString()).toBe("end\n");
			expect(env).toEqual({});
			// A second read (a restarted omp reusing the variables) finds nothing.
			expect(
				takeTopicsControlRequest({ ...env, HERDR_OMP_CONTROL_TOKEN_FILE: tokenFile, HERDR_OMP_CONTROL: "x" }),
			).toBe(undefined);
		} finally {
			tempDir.removeSync();
		}
	});
});
