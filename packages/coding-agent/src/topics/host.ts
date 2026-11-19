/**
 * Host-facing topics surface: what a host (RPC client, Herdr control socket)
 * may do with the owner's topics. A topic is a persistent worker in the
 * {@link VibeSessionRegistry.topics} scope of the top-level session; the host
 * talks to it directly, without a turn of the main conversation. While a host
 * lives, agents' `agent://<id>` messages to its topics run as topic turns
 * ({@link routeTopicMessage}).
 */
import { addIrcRoute } from "../irc/routes";
import { readRpcSubagentTranscript } from "../modes/rpc/rpc-subagents";
import type { RpcSubagentMessagesResult } from "../modes/rpc/rpc-types";
import type { ToolSession } from "../tools";
import { VibeSessionRegistry } from "../vibe/runtime";
import type { WorkerChange, WorkerSummary } from "../vibe/worker-scope";
import { cfgTopicsEnabled } from "./settings";
import { routeTopicMessage } from "./workers";

/** One topic as a host sees it. */
export type TopicSummary = WorkerSummary;

/** Outcome of sending to a topic: steered into a running turn, queued behind one, or a new turn. */
export interface TopicSendOutcome {
	id: string;
	mode: "steered" | "queued" | "turn";
}

export interface TopicsHost {
	list(): TopicSummary[];
	/** Start a topic with an owner-facing title and its first message. */
	create(args: { title: string; message: string; name?: string }): Promise<{ id: string }>;
	send(id: string, message: string): Promise<TopicSendOutcome>;
	/** Read the topic's own conversation from `fromByte` (incremental). */
	read(id: string, fromByte?: number): Promise<RpcSubagentMessagesResult>;
	close(id: string): Promise<void>;
	reopen(id: string): Promise<void>;
	rename(id: string, title: string): Promise<void>;
	/** Restore persisted topics of the current session (startup, session switch). */
	restore(): Promise<number>;
	onChange(listener: (change: WorkerChange) => void): () => void;
	/** Stop taking agents' messages to the topics; the topics themselves are kept. */
	dispose(): void;
}

/** Bind the topics registry to the top-level session's tool session. */
export function createTopicsHost(session: ToolSession): TopicsHost {
	const registry = VibeSessionRegistry.topics();
	const ownedBySession = (change: WorkerChange): boolean => {
		const scope = registry.ownerScope(session);
		return change.ownerId === scope.ownerId && change.parentSessionId === scope.parentSessionId;
	};
	const unroute = addIrcRoute((message, bus) => routeTopicMessage(registry, session, message, bus));
	return {
		list: () => registry.describe(session),
		async create({ title, message, name }) {
			const trimmed = title.trim();
			if (!trimmed) throw new Error("Topic title must not be empty.");
			const { id } = await registry.spawn(session, {
				cli: "good",
				name: name ?? trimmed,
				prompt: message,
				title: trimmed,
			});
			return { id };
		},
		async send(id, message) {
			const outcome = await registry.send(session, { session: id, message });
			return { id: outcome.id, mode: outcome.mode };
		},
		async read(id, fromByte) {
			const topic = registry.describe(session).find(entry => entry.id === id);
			if (!topic?.sessionFile) throw new Error(`Unknown topic: ${id}`);
			return readRpcSubagentTranscript(topic.sessionFile, fromByte);
		},
		close: id => registry.close(session, id),
		reopen: id => registry.reopen(session, id),
		rename: (id, title) => registry.rename(session, id, title),
		restore: () => registry.rehydrate(session),
		onChange(listener) {
			return registry.onChange(change => {
				if (ownedBySession(change)) listener(change);
			});
		},
		dispose: unroute,
	};
}

/** The session's topics host factory: only for a top-level session with `topics.enabled`. */
export function topicsHostFactory(session: ToolSession, topLevel: boolean): (() => TopicsHost) | undefined {
	return topLevel && cfgTopicsEnabled.get(session.settings) ? () => createTopicsHost(session) : undefined;
}

/**
 * A top-level session's topics host: created on first use and disposed with
 * the session, its persisted topics restored once per parent session file,
 * so a session switch restores the new session's topics.
 */
export class SessionTopics {
	readonly #create: () => TopicsHost;
	readonly #addDisposer: (dispose: () => void) => void;
	#host: TopicsHost | undefined;
	/** Parent session file whose persisted topics were last restored (`undefined` = never). */
	#restoredFor: string | null | undefined;

	constructor(create: () => TopicsHost, addDisposer: (dispose: () => void) => void) {
		this.#create = create;
		this.#addDisposer = addDisposer;
	}

	/** The host, with the topics of `sessionFile` restored. */
	async host(sessionFile: string | null): Promise<TopicsHost> {
		if (!this.#host) {
			const host = this.#create();
			this.#host = host;
			this.#addDisposer(() => host.dispose());
		}
		if (this.#restoredFor !== sessionFile) {
			await this.#host.restore();
			this.#restoredFor = sessionFile;
		}
		return this.#host;
	}
}
