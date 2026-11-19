/**
 * Recipient routes: registries that run some recipients' turns themselves
 * (topics) see each local IRC delivery before {@link IrcBus} hands it on. A
 * route is consulted by the bus that delivers locally (main's), whichever
 * thread the sender runs on.
 *
 * One route per owner (a top-level session's topics): registering again for
 * the same owner replaces its route, so a delivery runs each owner's route
 * once. Routes act only on their own owner's workers, so at most one takes
 * a message.
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

const routes = new Map<object, IrcRecipientRoute>();

/**
 * Route local deliveries through `route` first, as `owner`'s only route (it
 * replaces one `owner` registered before). Returns the unregister function,
 * which leaves a later replacement in place.
 */
export function setIrcRoute(owner: object, route: IrcRecipientRoute): () => void {
	routes.set(owner, route);
	return () => {
		if (routes.get(owner) === route) routes.delete(owner);
	};
}

/** Number of registered routes. Test-only. */
export function ircRouteCountForTests(): number {
	return routes.size;
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
	for (const route of routes.values()) {
		const decision = route(routed, bus);
		if (!decision) continue;
		if ("receipt" in decision) return { message: routed, receipt: decision.receipt };
		routed = decision.message;
	}
	return { message: routed };
}
