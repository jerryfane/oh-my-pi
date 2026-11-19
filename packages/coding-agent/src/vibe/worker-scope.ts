/**
 * Worker scopes: how a {@link VibeSessionRegistry} instance treats its
 * workers (Vibe by default; owner-facing topics with `topics/workers`), and
 * the views of workers it gives other modules.
 */
import type { SingleResult } from "@oh-my-pi/pi-tui/tools/task";
import type { VibeSessionState } from "@oh-my-pi/pi-tui/tools/vibe";
import type { WorkerScopeTag } from "./lifecycle";

/**
 * How one registry instance treats its workers. The Vibe director's registry
 * uses the defaults; the topics registry keeps owner-facing, permanent
 * workers whose results are read by the owner instead of the parent model.
 */
export interface WorkerScopePolicy {
	/** Persisted scope tag; events of other scopes are ignored. Absent = Vibe. */
	scope?: WorkerScopeTag;
	/** Deliver each settled turn result into the owner conversation as an async job result. */
	deliverResults: boolean;
	/**
	 * Run workers at the owner's own spawn depth, so helpers they start keep the
	 * nesting a top-level session has (`task.maxRecursionDepth`).
	 */
	ownerSpawnDepth: boolean;
	/** Extra context for a worker's turn from its owner's other workers (topics: their board). */
	boardFor?: (siblings: WorkerSummary[]) => string | undefined;
	/** Called once each turn settles (topics: answer the agents whose asks started it). */
	onTurnSettled?: (turn: SettledWorkerTurn) => Promise<void>;
}

/** Record state change observed by {@link VibeSessionRegistry.onChange} listeners. */
export interface WorkerChange {
	id: string;
	ownerId: string;
	parentSessionId: string;
	reason: "spawned" | "turn-started" | "turn-settled" | "closed" | "reopened" | "retitled" | "restored";
}

/** Owner-facing row for one worker (see {@link VibeSessionRegistry.describe}). */
export interface WorkerSummary {
	id: string;
	title: string;
	state: VibeSessionState | "closed";
	turns: number;
	queued: number;
	createdAt: number;
	lastActivityAt: number;
	/** Agent-written gist of the latest activity: intent, tool, or result preview. */
	lastActivity?: string;
	/** First line of the owner's latest message to the worker (spawn or send), when this process saw one. */
	ownerLine?: string;
	/** Absolute path of the worker's own conversation file, when persisted. */
	sessionFile?: string;
	/** When the turn in flight started; absent while no turn runs. */
	turnStartedAt?: number;
}

/** Who started a worker's turn and whose asks it answers (see {@link VibeSessionRegistry.worker}). */
export interface WorkerTurnOrigin {
	/** Agents whose messages started the turn; empty when the owner did. */
	startedBy: readonly string[];
	/** Asks the turn answers when it settles: who asked, and the IRC id of the ask. */
	askers: readonly { from: string; messageId: string }[];
}

/** A live worker as other modules see it (see {@link VibeSessionRegistry.worker}). */
export interface WorkerView {
	id: string;
	ownerId: string;
	closed: boolean;
	/** The turn in flight, and whether its session is streaming now. */
	turn?: WorkerTurnOrigin & { streaming: boolean };
}

/** A turn that just settled, for {@link WorkerScopePolicy.onTurnSettled}. */
export interface SettledWorkerTurn extends Pick<WorkerTurnOrigin, "askers"> {
	id: string;
	result?: SingleResult;
	/** Why a turn that produced no result failed. */
	error?: string;
}

/** A message held for a worker's next turn. */
export interface QueuedMessage {
	text: string;
	/** The agent that sent it; absent for the owner's own sends. */
	from?: string;
	/** IRC id of an ask, whose sender gets the turn's answer when it settles. */
	askId?: string;
	/** A progress note: rides along with the next turn, never starts one or counts as who started it. */
	note?: boolean;
}
