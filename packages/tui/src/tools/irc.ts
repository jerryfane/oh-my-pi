/** Peer-message types and roster constants shared by messaging, `wait`, and child prompts. */

/** Maximum live peer rows embedded in child prompts. */
export const DEFAULT_PEER_ROSTER_LIMIT = 32;

/** Serializable peer message retained in coordination result snapshots. */
export interface IrcMessage {
	id: string;
	/** Sender agent id. */
	from: string;
	/** Recipient agent id (resolved; "all" is expanded by the tool, not stored). */
	to: string;
	body: string;
	ts: number;
	/** Message id being answered. */
	replyTo?: string;
	/**
	 * Automated wake-turn relay of a woken subagent's stop output (task executor
	 * `relayWakeTurnOutput`). Relays are answers, never wake sources: the
	 * recipient's own wake-turn relay must skip them or two idle peers
	 * ping-pong forever.
	 */
	wakeRelay?: boolean;
	/**
	 * One leg of an `all` broadcast. Like relays, broadcasts never draw an
	 * automatic wake-turn reply: every woken idle peer would otherwise answer
	 * the broadcaster, mostly to say the message was not for it.
	 */
	broadcast?: boolean;
	/**
	 * Interim acknowledgement or progress on something the recipient asked
	 * (`agent://<id>?progress`): it neither answers that request nor asks one,
	 * so the recipient's `wait` keeps waiting for the actual reply.
	 */
	progress?: boolean;
	/**
	 * Set by delivery routing, never by a sender: an idle recipient takes the
	 * message into its history for its next turn instead of starting one (a
	 * progress note from a topic turn another agent's message started).
	 */
	noWake?: boolean;
}

/**
 * Delivery outcome for one peer recipient. `queued`: a busy topic holds the
 * message for a turn of its own once its current turn ends.
 */
export interface IrcDeliveryReceipt {
	to: string;
	outcome: "injected" | "woken" | "revived" | "queued" | "failed";
	error?: string;
}
/** Status ordering for peer rosters in child prompts. */
export const LIST_STATUS_ORDER: Record<string, number> = { running: 0, idle: 1, parked: 2 };
