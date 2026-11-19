/**
 * Local control socket for the owner's topics of an interactive session.
 *
 * A host that launched omp in a terminal pane (Herdr) cannot use stdio RPC, so
 * it passes `HERDR_OMP_CONTROL` (socket path inside a directory only it can
 * enter) and `HERDR_OMP_CONTROL_TOKEN` (per-launch secret). omp listens there
 * with owner-only permissions; a client must present the token in its first
 * frame. Frames are newline-delimited JSON, one object per line:
 *
 * - client `{"type":"hello","token":"…"}` → `{"type":"hello","protocol":1,"pid":…}`
 * - client `{"id":"1","type":"topics_list"}` (any `topics_*` RPC command) →
 *   `{"id":"1","type":"response","command":"topics_list","success":true,"data":…}`
 * - client `{"type":"subscribe"}` → `topic_update` frames `{"type":"topic_update","change":…}`
 *
 * The socket exposes only topics; it cannot prompt the main session.
 */
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import type { Socket, UnixSocketListener } from "bun";
import { logger } from "@oh-my-pi/pi-utils";
import { isRpcTopicsCommand, runRpcTopicsCommand } from "../modes/rpc/rpc-topics";
import type { RpcCommand } from "../modes/rpc/rpc-types";
import type { TopicsHost } from "./host";

/** Wire version announced in the hello reply; a host refuses majors it does not know. */
export const TOPICS_CONTROL_PROTOCOL = 1;
/** One frame larger than this closes the connection (a transcript read is the largest reply, not request). */
const MAX_REQUEST_BYTES = 1024 * 1024;

export interface TopicsControlOptions {
	path: string;
	token: string;
	/** Resolves the session's topics on demand (restores them after a session switch). */
	topics: () => Promise<TopicsHost | undefined>;
}

interface ConnectionState {
	buffer: string;
	authenticated: boolean;
	unsubscribe?: () => void;
	/** Serializes this connection's commands so replies keep request order. */
	queue: Promise<void>;
}

function constantTimeEqual(a: string, b: string): boolean {
	const left = Buffer.from(a);
	const right = Buffer.from(b);
	return left.length === right.length && crypto.timingSafeEqual(left, right);
}

/**
 * Start the control socket; returns a stop function. Refuses to start when the
 * path already exists (a stale or foreign socket is never replaced silently).
 */
export function startTopicsControl(options: TopicsControlOptions): () => void {
	if (!options.token) throw new Error("Topics control requires a non-empty token.");
	if (fs.existsSync(options.path)) throw new Error(`Topics control socket already exists: ${options.path}`);
	const previousUmask = process.umask(0o177);
	let server: UnixSocketListener<ConnectionState>;
	try {
		server = Bun.listen<ConnectionState>({
			unix: options.path,
			socket: {
				open(socket) {
					socket.data = { buffer: "", authenticated: false, queue: Promise.resolve() };
				},
				data(socket, chunk) {
					const state = socket.data;
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
						state.queue = state.queue.then(() => handleLine(socket, state, line, options));
					}
				},
				close(socket) {
					socket.data?.unsubscribe?.();
				},
				error(socket, error) {
					logger.debug("topics control connection error", { error: error.message });
					socket.data?.unsubscribe?.();
				},
			},
		});
	} finally {
		process.umask(previousUmask);
	}
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

type ControlSocket = Socket<ConnectionState>;

function send(socket: ControlSocket, frame: object): void {
	socket.write(`${JSON.stringify(frame)}\n`);
}

async function handleLine(
	socket: ControlSocket,
	state: ConnectionState,
	line: string,
	options: TopicsControlOptions,
): Promise<void> {
	let frame: unknown;
	try {
		frame = JSON.parse(line);
	} catch {
		send(socket, { type: "error", error: "Frame is not valid JSON." });
		return;
	}
	if (typeof frame !== "object" || frame === null || !("type" in frame) || typeof frame.type !== "string") {
		send(socket, { type: "error", error: "Frame needs a string `type`." });
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
	// Validated as a `topics_*` RPC command below; field checks happen in runRpcTopicsCommand.
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

/** Start the control socket when the launching host asked for one; returns a stop function or undefined. */
export function startTopicsControlFromEnv(topics: () => Promise<TopicsHost | undefined>): (() => void) | undefined {
	const path = process.env.HERDR_OMP_CONTROL?.trim();
	const token = process.env.HERDR_OMP_CONTROL_TOKEN?.trim();
	if (!path || !token) return undefined;
	try {
		return startTopicsControl({ path, token, topics });
	} catch (error) {
		logger.warn("topics control socket not started", {
			path,
			error: error instanceof Error ? error.message : String(error),
		});
		return undefined;
	}
}
