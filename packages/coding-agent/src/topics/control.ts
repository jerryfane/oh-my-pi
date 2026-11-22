/**
 * Local control socket for the owner's topics of an interactive session.
 *
 * A host that runs omp in a terminal pane (Herdr) cannot use stdio RPC. When
 * topics are enabled, omp opens its own socket at a fresh random path: inside
 * a fresh owner-only (0700) directory under `$XDG_RUNTIME_DIR` (else the temp
 * dir), or in the shared native relocation directory when that path does not
 * fit a Unix socket address ({@link ipcEndpoint}). A random per-launch secret
 * is held only in memory: never in a file,
 * an environment variable or a command line. In-process extensions read the
 * endpoint from {@link currentTopicsControl} (exported from the package root,
 * so `pi.pi.currentTopicsControl()`), prompted by a payload-free change signal
 * on the session event bus ({@link TOPICS_CONTROL_CHANNEL}); a host's extension
 * reports it to its host over the host's own API. Every launch, restart and
 * session switch gets a working endpoint. A client must present the secret in its first frame.
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
 * endpoint (extensions already run in-process with the agent's full rights),
 * so loading an extension grants this control too; emitting on the event bus
 * cannot change what they read.
 */
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { type IpcConnection, IpcServer, ipcEndpoint } from "@oh-my-pi/pi-natives";
import { logger } from "@oh-my-pi/pi-utils";
import type { Settings } from "../config/settings";
import { isRpcTopicsCommand, runRpcTopicsCommand, TOPICS_DISABLED, topicsFailure } from "../modes/rpc/rpc-topics";
import type { RpcCommand } from "../modes/rpc/rpc-types";
import type { TopicsHost } from "./host";
import { cfgTopicsEnabled } from "./settings";

/** Wire version announced in the hello reply; a host refuses majors it does not know. */
export const TOPICS_CONTROL_PROTOCOL = 1;
/** Shortest accepted secret; a weak token from a misconfigured host is refused. */
export const MIN_TOKEN_LENGTH = 32;
/** One request larger than this closes the connection. */
const MAX_REQUEST_BYTES = 1024 * 1024;
/** Reply bytes queued past the native write high-water mark beyond this close a client that stopped reading. */
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

interface ControlClient {
	connection: IpcConnection;
	authenticated: boolean;
	unsubscribe?: () => void;
	/** Serializes this connection's commands so replies keep request order. */
	queue: Promise<void>;
	/** Bytes sent since the native write queue passed its high-water mark; reset once it drains. */
	backlogBytes: number;
	helloTimer?: NodeJS.Timeout;
	closed: boolean;
}

function constantTimeEqual(a: string, b: string): boolean {
	const left = Buffer.from(a);
	const right = Buffer.from(b);
	return left.length === right.length && crypto.timingSafeEqual(left, right);
}

/** Whether a server accepts connections on `socketPath`; a dead socket left by a crash refuses them. */
async function socketAnswers(socketPath: string): Promise<boolean> {
	try {
		const probe = await Bun.connect({ unix: socketPath, socket: { data() {} } });
		probe.end();
		return true;
	} catch {
		return false;
	}
}

/**
 * Start the control socket at `options.path` (an {@link ipcEndpoint}); returns a stop function. The native
 * listener makes the socket owner-only (0600), drops peers of another user, refuses a path a live server
 * answers on (`EADDRINUSE`) and replaces a dead socket a crashed run left.
 */
export async function startTopicsControl(options: TopicsControlOptions): Promise<() => void> {
	if (options.token.length < MIN_TOKEN_LENGTH) {
		throw new Error(`Topics control token must be at least ${MIN_TOKEN_LENGTH} characters.`);
	}
	const helloTimeoutMs = options.helloTimeoutMs ?? DEFAULT_HELLO_TIMEOUT_MS;
	let unauthenticated = 0;
	let stopped = false;
	const clients = new Set<ControlClient>();
	const release = (client: ControlClient): void => {
		if (client.closed) return;
		client.closed = true;
		clients.delete(client);
		clearTimeout(client.helloTimer);
		if (!client.authenticated) unauthenticated--;
		client.unsubscribe?.();
	};
	const server = await IpcServer.listen(
		options.path,
		// A line longer than one request closes that connection natively.
		{ framing: "lines", maxFrameBytes: MAX_REQUEST_BYTES },
		connection => {
			if (stopped || unauthenticated >= MAX_UNAUTHENTICATED_CONNECTIONS) {
				connection.destroy();
				return;
			}
			unauthenticated++;
			const client: ControlClient = {
				connection,
				authenticated: false,
				queue: Promise.resolve(),
				backlogBytes: 0,
				closed: false,
			};
			clients.add(client);
			// Also bounds a refused client that keeps its end open after our `end()`.
			client.helloTimer = setTimeout(() => {
				if (!client.authenticated) connection.destroy();
			}, helloTimeoutMs);
			connection.start(
				received => {
					const line = received.trim();
					if (!line || client.closed) return;
					client.queue = client.queue
						.then(() =>
							handleLine(client, line, options, () => {
								unauthenticated--;
								clearTimeout(client.helloTimer);
							}),
						)
						// A handler failure must never stall this connection's later commands.
						.catch(error => logger.warn("topics control: command failed", { error: String(error) }));
				},
				reason => {
					if (reason) logger.debug("topics control connection closed", { reason });
					release(client);
				},
			);
		},
	);
	// The socket and the native listener's bind lock; a pipe (Windows) leaves nothing on disk.
	const files = process.platform === "win32" ? [] : [options.path, `${options.path}.bind.lock`];
	return () => {
		if (stopped) return;
		stopped = true;
		for (const client of clients) client.connection.destroy();
		// Unlinked now, so no client can reach the path while `close` finishes; `close` only unlinks
		// a socket that is still its own inode.
		for (const file of files) {
			try {
				fs.rmSync(file, { force: true });
			} catch (error) {
				logger.warn("topics control: failed to remove socket", { error: String(error) });
			}
		}
		server.close().catch(error => logger.warn("topics control: failed to close", { error: String(error) }));
	};
}

/** Send one frame; close a client that stopped reading. */
function send(client: ControlClient, frame: object): void {
	if (client.closed) return;
	const line = JSON.stringify(frame);
	if (client.connection.send(line)) return;
	const draining = client.backlogBytes > 0;
	client.backlogBytes += Buffer.byteLength(line) + 1;
	if (client.backlogBytes > MAX_PENDING_OUTPUT_BYTES) {
		logger.warn("topics control: closing a client that stopped reading");
		client.connection.destroy();
		return;
	}
	if (!draining) {
		void client.connection.drained().then(() => {
			client.backlogBytes = 0;
		});
	}
}

async function handleLine(
	client: ControlClient,
	line: string,
	options: TopicsControlOptions,
	onAuthenticated: () => void,
): Promise<void> {
	if (client.closed) return;
	let frame: unknown;
	try {
		frame = JSON.parse(line);
	} catch {
		frame = undefined;
	}
	if (typeof frame !== "object" || frame === null || !("type" in frame) || typeof frame.type !== "string") {
		send(client, { type: "error", error: "Frame must be a JSON object with a string `type`." });
		if (!client.authenticated) client.connection.end();
		return;
	}
	const id = "id" in frame && typeof frame.id === "string" ? frame.id : undefined;

	if (!client.authenticated) {
		const token = "token" in frame && typeof frame.token === "string" ? frame.token : "";
		if (frame.type !== "hello" || !constantTimeEqual(token, options.token)) {
			send(client, { type: "error", error: "Unauthorized." });
			client.connection.end();
			return;
		}
		client.authenticated = true;
		onAuthenticated();
		send(client, { type: "hello", protocol: TOPICS_CONTROL_PROTOCOL, pid: process.pid });
		return;
	}

	// Any failure (resolving the host included) answers this command; the connection keeps working.
	try {
		await runAuthenticated(client, frame.type, id, frame, options);
	} catch (error) {
		send(client, { id, type: "response", command: frame.type, success: false, ...topicsFailure(error) });
	}
}

async function runAuthenticated(
	client: ControlClient,
	type: string,
	id: string | undefined,
	frame: object,
	options: TopicsControlOptions,
): Promise<void> {
	const topics = await options.topics();
	if (!topics) {
		send(client, {
			id,
			type: "response",
			command: type,
			success: false,
			error: "Topics are disabled.",
			code: TOPICS_DISABLED.code,
		});
		return;
	}
	if (type === "subscribe") {
		client.unsubscribe ??= topics.onChange(change => send(client, { type: "topic_update", change }));
		send(client, { id, type: "response", command: "subscribe", success: true });
		return;
	}
	// Narrowed to a `topics_*` command below; runRpcTopicsCommand validates each field.
	const command = frame as RpcCommand;
	if (!isRpcTopicsCommand(command)) {
		send(client, { id, type: "response", command: type, success: false, error: "Unknown command." });
		return;
	}
	const data = await runRpcTopicsCommand(topics, command);
	send(client, { id, type: "response", command: command.type, success: true, ...(data ? { data } : {}) });
}

/**
 * Event-bus channel signalling that the endpoint changed (opened or closed).
 * Its payload is only {@link TopicsControlChanged}: any extension can emit on
 * the bus, so the endpoint itself is read from {@link currentTopicsControl}.
 */
export const TOPICS_CONTROL_CHANNEL = "topics:control";
/** Extensions emit this to have the change signal repeated (e.g. after they loaded late). */
export const TOPICS_CONTROL_REQUEST_CHANNEL = "topics:control:request";

/** The only payload omp emits on {@link TOPICS_CONTROL_CHANNEL}. */
export interface TopicsControlChanged {
	changed: true;
}

/** The open endpoint, as {@link currentTopicsControl} returns it. */
export interface TopicsControlEndpoint {
	socketPath: string;
	token: string;
	protocol: number;
	pid: number;
}

/** The event bus surface the change signal is emitted on (the session's extension bus). */
export interface TopicsControlBus {
	emit(channel: string, data: unknown): void;
	on(channel: string, handler: (data: unknown) => void): () => void;
}

/** Set only by {@link offerTopicsControl}; nothing on the event bus can change it. */
let currentEndpoint: TopicsControlEndpoint | undefined;

/**
 * The control endpoint omp opened, or `undefined` while none is open. Exported
 * from the package root, so an extension reads it through its injected API
 * (`pi.pi.currentTopicsControl()`) when {@link TOPICS_CONTROL_CHANNEL} fires.
 */
export function currentTopicsControl(): TopicsControlEndpoint | undefined {
	return currentEndpoint;
}

/** Directory name prefix of a control directory; a random suffix follows. */
const CONTROL_DIR_PREFIX = "omp-topics-";
/** {@link ipcEndpoint} name of the socket: `<dir>/control.sock` unless relocated. */
const CONTROL_ENDPOINT_NAME = "control";
const CONTROL_SOCKET_NAME = `${CONTROL_ENDPOINT_NAME}.sock`;

/**
 * Remove control directories that crashed runs left in `runtimeDir`: our own
 * `omp-topics-*` directories whose `control.sock` refuses connections. One
 * that answers belongs to a live omp and is kept, as is one without a socket
 * (another omp may be between creating it and listening).
 */
async function removeStaleControlDirs(runtimeDir: string): Promise<void> {
	const uid = process.getuid?.();
	if (uid === undefined) return;
	let names: string[];
	try {
		names = fs.readdirSync(runtimeDir);
	} catch {
		return;
	}
	for (const name of names) {
		if (!name.startsWith(CONTROL_DIR_PREFIX)) continue;
		const dir = path.join(runtimeDir, name);
		const socketPath = path.join(dir, CONTROL_SOCKET_NAME);
		try {
			const dirStat = fs.lstatSync(dir);
			if (!dirStat.isDirectory() || dirStat.uid !== uid) continue;
			if (!fs.lstatSync(socketPath).isSocket()) continue;
		} catch {
			continue;
		}
		if (await socketAnswers(socketPath)) continue;
		fs.rmSync(dir, { recursive: true, force: true });
	}
}

/**
 * Create the fresh control directory `dir` owner-only. `mkdir`'s mode is masked by the umask (a restrictive
 * one leaves no owner access at all), so the mode is set explicitly and then verified.
 */
function createPrivateDir(dir: string): void {
	fs.mkdirSync(dir, { mode: 0o700 });
	fs.chmodSync(dir, 0o700);
	const stat = fs.lstatSync(dir);
	const uid = process.getuid?.();
	if (!stat.isDirectory() || (uid !== undefined && stat.uid !== uid) || (stat.mode & 0o777) !== 0o700) {
		throw new Error(`${dir} is not an owner-only directory of this user (mode ${(stat.mode & 0o777).toString(8)})`);
	}
}

export interface OfferTopicsControlOptions {
	/** Tell the user, once, that the control socket is unavailable and why. */
	notify?: (message: string) => void;
}

/**
 * Open the control socket at a fresh random path with a random secret, after
 * removing what crashed runs left behind ({@link removeStaleControlDirs}). The
 * endpoint is `ipcEndpoint(<runtimeDir>/omp-topics-<random>, "control")`: the
 * socket inside a fresh owner-only directory, or, when that path does not fit
 * a Unix socket address, a unique socket in the native private relocation
 * directory (whose placement and checks the native listener owns). It is
 * published to {@link currentTopicsControl}; `bus` only hears that it
 * changed: now, on every {@link TOPICS_CONTROL_REQUEST_CHANNEL}, and when
 * stopped. Never rejects: any failure is logged and told to the user once
 * (`notify`), and the session runs without host control. Returns a stop
 * function, or undefined when it could not start.
 */
export async function offerTopicsControl(
	bus: TopicsControlBus,
	topics: () => Promise<TopicsHost | undefined>,
	/** `$XDG_RUNTIME_DIR` (user-private, cleared at logout) when set, else the temp dir. */
	runtimeDir: string = process.env.XDG_RUNTIME_DIR || os.tmpdir(),
	options: OfferTopicsControlOptions = {},
): Promise<(() => void) | undefined> {
	const dir = path.join(runtimeDir, `${CONTROL_DIR_PREFIX}${crypto.randomBytes(9).toString("base64url")}`);
	let socketPath: string | undefined;
	let createdDir = false;
	const changed: TopicsControlChanged = { changed: true };
	try {
		socketPath = ipcEndpoint(dir, CONTROL_ENDPOINT_NAME);
		// Relocated (or a Windows pipe): nothing of ours to create next to the endpoint.
		const ownsDir = path.dirname(socketPath) === dir;
		await removeStaleControlDirs(runtimeDir);
		if (ownsDir) {
			createdDir = true;
			createPrivateDir(dir);
		}
		const endpoint: TopicsControlEndpoint = {
			socketPath,
			token: crypto.randomBytes(32).toString("base64url"),
			protocol: TOPICS_CONTROL_PROTOCOL,
			pid: process.pid,
		};
		const stopSocket = await startTopicsControl({ path: socketPath, token: endpoint.token, topics });
		currentEndpoint = endpoint;
		const unsubscribe = bus.on(TOPICS_CONTROL_REQUEST_CHANNEL, () => bus.emit(TOPICS_CONTROL_CHANNEL, changed));
		bus.emit(TOPICS_CONTROL_CHANNEL, changed);
		return () => {
			unsubscribe();
			stopSocket();
			if (currentEndpoint === endpoint) currentEndpoint = undefined;
			bus.emit(TOPICS_CONTROL_CHANNEL, changed);
			if (ownsDir) fs.rmSync(dir, { recursive: true, force: true });
		};
	} catch (error) {
		const where = socketPath ?? dir;
		const message = `Topics control socket unavailable at ${where}: ${error instanceof Error ? error.message : String(error)}. Topics still work in omp; only a host's (Herdr's) control of them is off.`;
		logger.error(message);
		options.notify?.(message);
		try {
			if (createdDir) fs.rmSync(dir, { recursive: true, force: true });
			// A failed listen may leave the native bind lock next to the (unique) endpoint.
			if (socketPath && process.platform !== "win32") fs.rmSync(`${socketPath}.bind.lock`, { force: true });
		} catch (cleanupError) {
			logger.warn("topics control: cleanup after a failed start failed", { error: String(cleanupError) });
		}
		return undefined;
	}
}

/** With `topics.enabled`, offer the control endpoint for `session`'s lifetime (interactive sessions). */
export async function offerSessionTopicsControl(
	session: { getTopicsHost(): Promise<TopicsHost | undefined>; addDisposer(dispose: () => void): void },
	bus: TopicsControlBus,
	settings: Settings,
	/** Shows the user a startup notice (the control socket could not be opened). */
	notify?: (message: string) => void,
): Promise<void> {
	if (!cfgTopicsEnabled.get(settings)) return;
	const stop = await offerTopicsControl(bus, () => session.getTopicsHost(), undefined, { notify });
	if (stop) session.addDisposer(stop);
}
