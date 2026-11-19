/**
 * Contract: the topics control socket is an owner-only door that a host
 * opens with its per-launch token and that reaches topics, nothing else.
 * Extensions read the endpoint from omp's own state, never from the event
 * bus, which only says that it changed; startup clears what crashed runs left.
 */
import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as fs from "node:fs";
import * as path from "node:path";
import { ExtensionRuntime, loadExtensionFromFactory } from "@oh-my-pi/pi-coding-agent/extensibility/extensions/loader";
import type { RpcSubagentMessagesResult } from "@oh-my-pi/pi-coding-agent/modes/rpc/rpc-types";
import {
	currentTopicsControl,
	MIN_TOKEN_LENGTH,
	offerTopicsControl,
	startTopicsControl,
	TOPICS_CONTROL_CHANNEL,
	TOPICS_CONTROL_PROTOCOL,
	TOPICS_CONTROL_REQUEST_CHANNEL,
	type TopicsControlEndpoint,
} from "@oh-my-pi/pi-coding-agent/topics/control";
import { EventBus } from "@oh-my-pi/pi-coding-agent/utils/event-bus";
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
		dispose: () => {},
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

	it("refuses a mistyped topics command instead of reporting success", async () => {
		const client = await connect(socketPath);
		client.send({ type: "hello", token: TOKEN });
		await client.next();
		client.send({ id: "typo", type: "topics_clsoe", topicId: "Diokan" });
		expect(await client.next()).toMatchObject({ id: "typo", success: false, error: "Unknown command." });
		client.end();
	});

	it("answers a command whose host lookup throws, and keeps serving the connection", async () => {
		stop?.();
		let failures = 1;
		const host = fakeHost(sent);
		stop = await startTopicsControl({
			path: socketPath,
			token: TOKEN,
			topics: async () => {
				if (failures-- > 0) throw new Error("restore failed");
				return host;
			},
		});
		const client = await connect(socketPath);
		client.send({ type: "hello", token: TOKEN });
		await client.next();
		client.send({ id: "1", type: "topics_list" });
		expect(await client.next()).toEqual({
			id: "1",
			type: "response",
			command: "topics_list",
			success: false,
			error: "restore failed",
		});
		client.send({ id: "2", type: "topics_list" });
		expect(await client.next()).toMatchObject({ id: "2", success: true, data: { topics: [{ id: "Diokan" }] } });
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

/** Leave a socket inode at `socketPath` with nobody listening, as a SIGKILLed run does. */
function deadSocket(socketPath: string): void {
	const crashed = Bun.listen({ unix: socketPath, socket: { data() {} } });
	fs.linkSync(socketPath, `${socketPath}.keep`);
	crashed.stop(true);
	fs.renameSync(`${socketPath}.keep`, socketPath);
}

describe("topics control endpoint for extensions", () => {
	let tempDir: TempDir;
	const stops: Array<() => void> = [];

	beforeEach(() => {
		tempDir = TempDir.createSync("@omp-topics-offer-");
	});

	afterEach(() => {
		for (const stop of stops.splice(0)) stop();
		tempDir.removeSync();
	});

	async function offer(bus: EventBus, runtimeDir?: string): Promise<TopicsControlEndpoint> {
		const stop = await offerTopicsControl(bus, async () => fakeHost([]), runtimeDir);
		if (!stop) throw new Error("control socket did not start");
		stops.push(stop);
		const endpoint = currentTopicsControl();
		if (!endpoint) throw new Error("no endpoint published");
		return endpoint;
	}

	it("publishes a fresh private endpoint, signals changes without the secret, and withdraws it on stop", async () => {
		const bus = new EventBus();
		const signals: unknown[] = [];
		bus.on(TOPICS_CONTROL_CHANNEL, data => signals.push(data));
		const endpoint = await offer(bus, tempDir.path());

		expect(endpoint.protocol).toBe(TOPICS_CONTROL_PROTOCOL);
		expect(endpoint.token.length).toBeGreaterThanOrEqual(MIN_TOKEN_LENGTH);
		expect(fs.statSync(path.dirname(endpoint.socketPath)).mode & 0o777).toBe(0o700);
		// The secret lives only in memory: nothing under the runtime dir contains it.
		for (const entry of fs.readdirSync(tempDir.path(), { recursive: true })) {
			const file = path.join(tempDir.path(), String(entry));
			if (fs.statSync(file).isFile()) expect(fs.readFileSync(file, "utf8")).not.toContain(endpoint.token);
		}
		// The bus only says "changed", now and when a late extension asks again.
		bus.emit(TOPICS_CONTROL_REQUEST_CHANNEL, undefined);
		expect(signals).toEqual([{ changed: true }, { changed: true }]);
		expect(JSON.stringify(signals)).not.toContain(endpoint.token);
		const client = await connect(endpoint.socketPath);
		client.send({ type: "hello", token: endpoint.token });
		expect(await client.next()).toMatchObject({ type: "hello" });
		client.end();

		stops.shift()?.();
		expect(currentTopicsControl()).toBeUndefined();
		expect(signals.at(-1)).toEqual({ changed: true });
		expect(fs.existsSync(path.dirname(endpoint.socketPath))).toBe(false);
	});

	it("gives every launch its own path and secret", async () => {
		const first = await offer(new EventBus(), tempDir.path());
		const second = await offer(new EventBus(), tempDir.path());
		expect(first.socketPath).not.toBe(second.socketPath);
		expect(first.token).not.toBe(second.token);
	});

	it("reaches an extension through its injected pi API, and a fake emission on the channel changes nothing", async () => {
		const bus = new EventBus();
		const seen: Array<TopicsControlEndpoint | undefined> = [];
		let forge: (() => void) | undefined;
		await loadExtensionFromFactory(
			pi => {
				pi.events.on(TOPICS_CONTROL_CHANNEL, () => seen.push(pi.pi.currentTopicsControl()));
				forge = () =>
					pi.events.emit(TOPICS_CONTROL_CHANNEL, {
						socketPath: path.join(tempDir.path(), "evil.sock"),
						token: "x".repeat(MIN_TOKEN_LENGTH),
						protocol: TOPICS_CONTROL_PROTOCOL,
						pid: 1,
					});
			},
			tempDir.path(),
			bus,
			new ExtensionRuntime(),
			"host-extension",
		);
		const endpoint = await offer(bus, tempDir.path());
		expect(seen).toEqual([endpoint]);

		forge?.();
		expect(currentTopicsControl()).toEqual(endpoint);
		expect(seen.at(-1)).toEqual(endpoint);
	});

	it("opens under $XDG_RUNTIME_DIR when it is set", async () => {
		const previous = process.env.XDG_RUNTIME_DIR;
		process.env.XDG_RUNTIME_DIR = tempDir.path();
		try {
			const endpoint = await offer(new EventBus());
			expect(path.dirname(path.dirname(endpoint.socketPath))).toBe(tempDir.path());
		} finally {
			if (previous === undefined) delete process.env.XDG_RUNTIME_DIR;
			else process.env.XDG_RUNTIME_DIR = previous;
		}
	});

	it("removes control dirs whose socket is dead at startup, and never one that answers", async () => {
		const dirOf = (name: string) => {
			const dir = path.join(tempDir.path(), name);
			fs.mkdirSync(dir, { mode: 0o700 });
			return dir;
		};
		const stale = dirOf("omp-topics-stale");
		deadSocket(path.join(stale, "control.sock"));
		const live = dirOf("omp-topics-live");
		const listener = Bun.listen({ unix: path.join(live, "control.sock"), socket: { data() {} } });
		const starting = dirOf("omp-topics-starting");
		const other = dirOf("other-dir");
		deadSocket(path.join(other, "control.sock"));
		try {
			const endpoint = await offer(new EventBus(), tempDir.path());
			expect(fs.existsSync(stale)).toBe(false);
			expect(fs.existsSync(path.join(live, "control.sock"))).toBe(true);
			expect(fs.existsSync(starting)).toBe(true);
			expect(fs.existsSync(other)).toBe(true);
			expect(fs.existsSync(endpoint.socketPath)).toBe(true);
		} finally {
			listener.stop(true);
		}
	});
});
