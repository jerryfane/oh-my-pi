/**
 * Local control socket for the owner's topics of an interactive session.
 *
 * A host that runs omp in a terminal pane (Herdr) cannot use stdio RPC. When
 * topics are enabled, omp opens its own socket at a fresh random path inside
 * an owner-only (0700) directory, with a random per-launch secret held only in
 * memory: never in a file, an environment variable or a command line. It
 * offers the endpoint to in-process extensions on the session event bus
 * ({@link TOPICS_CONTROL_CHANNEL}); a host's extension reports it to its host
 * over the host's own API. Every launch, restart and session switch gets a
 * working endpoint. A client must present the secret in its first frame.
 * Frames are newline-delimited JSON, one per line:
 *
 * - client `{"type":"hello","token":"…"}` → `{"type":"hello","protocol":1,"pid":…}`
 * - client `{"id":"1","type":"topics_list"}` (any `topics_*` RPC command) →
 *   `{"id":"1","type":"response","command":"topics_list","success":true,"data":…}`
 * - client `{"type":"subscribe"}` → `topic_update` frames `{"type":"topic_update","change":…}`
 *
 * Privilege: the socket cannot prompt the main session, but `topics_send`
 * runs a topic worker with the agent's full tools, so holding the secret is
 * equivalent to controlling the agent. Every loaded extension can read the
 * endpoint from the event bus (extensions already run in-process with the
 * agent's full rights), so loading an extension grants this control too.
 */
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import type { Socket, UnixSocketListener } from "bun";
import { logger } from "@oh-my-pi/pi-utils";
import { isRpcTopicsCommand, runRpcTopicsCommand } from "../modes/rpc/rpc-topics";
import type { RpcCommand } from "../modes/rpc/rpc-types";
import type { TopicsHost } from "./host";

/** Wire version announced in the hello reply; a host refuses majors it does not know. */
export const TOPICS_CONTROL_PROTOCOL = 1;
/** Shortest accepted secret; a weak token from a misconfigured host is refused. */
export const MIN_TOKEN_LENGTH = 32;
/** One request larger than this closes the connection. */
const MAX_REQUEST_BYTES = 1024 * 1024;
/** Unsent reply bytes beyond this close a client that stopped reading. */
const MAX_PENDING_OUTPUT_BYTES = 64 * 1024 * 1024;
const DEFAULT_HELLO_TIMEOUT_MS = 5_000;
const MAX_UNAUTHENTICATED_CONNECTIONS = 8;

export interface TopicsControlOptions {
	path: string;
	token: string;
	/** Resolves the session's topics on demand (restores them after a session switch). */
	topics: () => Promise<TopicsHost | undefined>;
	/** Close a connection that has not authenticated within this time. */
	helloTimeoutMs?: number;
}

interface ConnectionState {
	buffer: string;
	authenticated: boolean;
	unsubscribe?: () => void;
	/** Serializes this connection's commands so replies keep request order. */
	queue: Promise<void>;
	/** Bytes the kernel did not accept yet; flushed on drain. */
	pending: Buffer[];
	pendingBytes: number;
	helloTimer?: NodeJS.Timeout;
	closed: boolean;
}

type ControlSocket = Socket<ConnectionState>;

function constantTimeEqual(a: string, b: string): boolean {
	const left = Buffer.from(a);
	const right = Buffer.from(b);
	return left.length === right.length && crypto.timingSafeEqual(left, right);
}

/** Write what the socket accepts now, keep the rest for drain; close a client that stops reading. */
function flush(socket: ControlSocket): void {
	const state = socket.data;
	while (state.pending.length > 0) {
		const chunk = state.pending[0];
		const written = socket.write(chunk);
		if (written < 0) return;
		state.pendingBytes -= written;
		if (written < chunk.length) {
			state.pending[0] = chunk.subarray(written);
			return;
		}
		state.pending.shift();
	}
}

function send(socket: ControlSocket, frame: object): void {
	const state = socket.data;
	if (state.closed) return;
	const bytes = Buffer.from(`${JSON.stringify(frame)}\n`);
	state.pending.push(bytes);
	state.pendingBytes += bytes.length;
	if (state.pendingBytes > MAX_PENDING_OUTPUT_BYTES) {
		logger.warn("topics control: closing a client that stopped reading");
		socket.end();
		return;
	}
	flush(socket);
}

/**
 * Remove a socket left by a crashed run: only a socket that refuses connections
 * is removed; anything else at the path (a live server, a regular file) is kept
 * and startup fails.
 */
async function clearStaleSocket(socketPath: string): Promise<void> {
	let stat: fs.Stats;
	try {
		stat = fs.lstatSync(socketPath);
	} catch {
		return;
	}
	if (!stat.isSocket()) throw new Error(`Topics control path exists and is not a socket: ${socketPath}`);
	try {
		const probe = await Bun.connect({ unix: socketPath, socket: { data() {} } });
		probe.end();
	} catch {
		fs.rmSync(socketPath, { force: true });
		return;
	}
	throw new Error(`Topics control socket is in use: ${socketPath}`);
}

/** Start the control socket; returns a stop function. */
export async function startTopicsControl(options: TopicsControlOptions): Promise<() => void> {
	if (options.token.length < MIN_TOKEN_LENGTH) {
		throw new Error(`Topics control token must be at least ${MIN_TOKEN_LENGTH} characters.`);
	}
	await clearStaleSocket(options.path);
	const helloTimeoutMs = options.helloTimeoutMs ?? DEFAULT_HELLO_TIMEOUT_MS;
	let unauthenticated = 0;
	const release = (state: ConnectionState): void => {
		if (state.closed) return;
		state.closed = true;
		clearTimeout(state.helloTimer);
		if (!state.authenticated) unauthenticated--;
		state.unsubscribe?.();
	};
	const server: UnixSocketListener<ConnectionState> = Bun.listen<ConnectionState>({
		unix: options.path,
		socket: {
			open(socket) {
				const state: ConnectionState = {
					buffer: "",
					authenticated: false,
					queue: Promise.resolve(),
					pending: [],
					pendingBytes: 0,
					closed: false,
				};
				socket.data = state;
				if (unauthenticated >= MAX_UNAUTHENTICATED_CONNECTIONS) {
					state.closed = true;
					socket.end();
					return;
				}
				unauthenticated++;
				state.helloTimer = setTimeout(() => {
					if (!state.authenticated) socket.end();
				}, helloTimeoutMs);
			},
			data(socket, chunk) {
				const state = socket.data;
				if (state.closed) return;
				state.buffer += chunk.toString("utf8");
				if (state.buffer.length > MAX_REQUEST_BYTES) {
					socket.end();
					return;
				}
				let newline = state.buffer.indexOf("\n");
				while (newline >= 0) {
					const line = state.buffer.slice(0, newline).trim();
					state.buffer = state.buffer.slice(newline + 1);
					newline = state.buffer.indexOf("\n");
					if (!line) continue;
					state.queue = state.queue.then(() =>
						handleLine(socket, state, line, options, () => {
							unauthenticated--;
							clearTimeout(state.helloTimer);
						}),
					);
				}
			},
			drain(socket) {
				flush(socket);
			},
			close(socket) {
				if (socket.data) release(socket.data);
			},
			error(socket, error) {
				logger.debug("topics control connection error", { error: error.message });
				if (socket.data) release(socket.data);
			},
		},
	});
	// Inside the host's 0700 directory the brief default-mode window is unreachable; tighten anyway.
	fs.chmodSync(options.path, 0o600);
	return () => {
		server.stop(true);
		try {
			fs.rmSync(options.path, { force: true });
		} catch (error) {
			logger.warn("topics control: failed to remove socket", { error: String(error) });
		}
	};
}

async function handleLine(
	socket: ControlSocket,
	state: ConnectionState,
	line: string,
	options: TopicsControlOptions,
	onAuthenticated: () => void,
): Promise<void> {
	if (state.closed) return;
	let frame: unknown;
	try {
		frame = JSON.parse(line);
	} catch {
		frame = undefined;
	}
	if (typeof frame !== "object" || frame === null || !("type" in frame) || typeof frame.type !== "string") {
		send(socket, { type: "error", error: "Frame must be a JSON object with a string `type`." });
		if (!state.authenticated) socket.end();
		return;
	}
	const id = "id" in frame && typeof frame.id === "string" ? frame.id : undefined;

	if (!state.authenticated) {
		const token = "token" in frame && typeof frame.token === "string" ? frame.token : "";
		if (frame.type !== "hello" || !constantTimeEqual(token, options.token)) {
			send(socket, { type: "error", error: "Unauthorized." });
			socket.end();
			return;
		}
		state.authenticated = true;
		onAuthenticated();
		send(socket, { type: "hello", protocol: TOPICS_CONTROL_PROTOCOL, pid: process.pid });
		return;
	}

	const topics = await options.topics();
	if (!topics) {
		send(socket, { id, type: "response", command: frame.type, success: false, error: "Topics are disabled." });
		return;
	}
	if (frame.type === "subscribe") {
		state.unsubscribe ??= topics.onChange(change => send(socket, { type: "topic_update", change }));
		send(socket, { id, type: "response", command: "subscribe", success: true });
		return;
	}
	// Narrowed to a `topics_*` command below; runRpcTopicsCommand validates each field.
	const command = frame as RpcCommand;
	if (!isRpcTopicsCommand(command)) {
		send(socket, { id, type: "response", command: frame.type, success: false, error: "Unknown command." });
		return;
	}
	try {
		const data = await runRpcTopicsCommand(topics, command);
		send(socket, { id, type: "response", command: command.type, success: true, ...(data ? { data } : {}) });
	} catch (error) {
		send(socket, {
			id,
			type: "response",
			command: command.type,
			success: false,
			error: error instanceof Error ? error.message : String(error),
		});
	}
}

/** Event-bus channel carrying the endpoint to extensions; see {@link offerTopicsControl}. */
export const TOPICS_CONTROL_CHANNEL = "topics:control";
/** Extensions emit this to ask for the current endpoint (e.g. after they loaded late). */
export const TOPICS_CONTROL_REQUEST_CHANNEL = "topics:control:request";

/** What an extension receives on {@link TOPICS_CONTROL_CHANNEL}: `null` once the socket closed. */
export interface TopicsControlEndpoint {
	socketPath: string;
	token: string;
	protocol: number;
	pid: number;
}

/** The event bus surface the endpoint is offered on (the session's extension bus). */
export interface TopicsControlBus {
	emit(channel: string, data: unknown): void;
	on(channel: string, handler: (data: unknown) => void): () => void;
}

/**
 * Open the control socket at a fresh random path with a random secret and
 * offer it on `bus`: once now, again on every {@link TOPICS_CONTROL_REQUEST_CHANNEL},
 * and `null` when stopped. Returns a stop function, or undefined when it could
 * not start.
 */
export async function offerTopicsControl(
	bus: TopicsControlBus,
	topics: () => Promise<TopicsHost | undefined>,
	runtimeDir: string = os.tmpdir(),
): Promise<(() => void) | undefined> {
	let dir: string | undefined;
	try {
		dir = fs.mkdtempSync(path.join(runtimeDir, "omp-topics-"));
		fs.chmodSync(dir, 0o700);
		const endpoint: TopicsControlEndpoint = {
			socketPath: path.join(dir, "control.sock"),
			token: crypto.randomBytes(32).toString("base64url"),
			protocol: TOPICS_CONTROL_PROTOCOL,
			pid: process.pid,
		};
		const stopSocket = await startTopicsControl({ path: endpoint.socketPath, token: endpoint.token, topics });
		const unsubscribe = bus.on(TOPICS_CONTROL_REQUEST_CHANNEL, () => bus.emit(TOPICS_CONTROL_CHANNEL, endpoint));
		bus.emit(TOPICS_CONTROL_CHANNEL, endpoint);
		const ownedDir = dir;
		return () => {
			unsubscribe();
			stopSocket();
			bus.emit(TOPICS_CONTROL_CHANNEL, null);
			fs.rmSync(ownedDir, { recursive: true, force: true });
		};
	} catch (error) {
		logger.warn("topics control socket not started", {
			error: error instanceof Error ? error.message : String(error),
		});
		if (dir) fs.rmSync(dir, { recursive: true, force: true });
		return undefined;
	}
}
