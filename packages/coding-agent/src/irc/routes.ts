/**
 * Recipient routes: registries that run some recipients' turns themselves
 * (topics) see each local IRC delivery before {@link IrcBus} hands it on. A
 * route consulted by the bus that delivers locally (main's), whichever thread
 * the sender runs on.
 */
import type { IrcDeliveryReceipt, IrcMessage } from "@oh-my-pi/pi-tui/tools/irc";
import type { IrcBusApi } from "./bus";

/**
 * What a {@link IrcRecipientRoute} decides: take delivery (`{ receipt }`), have
 * ordinary delivery carry a rewritten `{ message }` instead, or leave the
 * message as it is (`undefined`).
 */
export type IrcRouteDecision = { receipt: IrcDeliveryReceipt } | { message: IrcMessage } | undefined;

/** One route; `bus` is the delivering bus, for its send bookkeeping. */
export type IrcRecipientRoute = (message: IrcMessage, bus: IrcBusApi) => IrcRouteDecision;

const routes = new Set<IrcRecipientRoute>();

/** Route local deliveries through `route` first; returns the unregister function. */
export function addIrcRoute(route: IrcRecipientRoute): () => void {
	routes.add(route);
	return () => routes.delete(route);
}

/**
 * Run `message` through every route: the receipt of the route that took it,
 * if one did, and the message ordinary delivery should carry otherwise.
 */
export function routeIrcMessage(
	message: IrcMessage,
	bus: IrcBusApi,
): { message: IrcMessage; receipt?: IrcDeliveryReceipt } {
	let routed = message;
	for (const route of routes) {
		const decision = route(routed, bus);
		if (!decision) continue;
		if ("receipt" in decision) return { message: routed, receipt: decision.receipt };
		routed = decision.message;
	}
	return { message: routed };
}
