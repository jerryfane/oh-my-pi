import { describe, expect, it, spyOn } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { Process, ProcessStatus, processInfo } from "@oh-my-pi/pi-natives";
import {
	AbortError,
	ChildProcess,
	createLinuxSubreaperScript,
	exec,
	NonZeroExitError,
	spawn,
	TimeoutError,
} from "@oh-my-pi/pi-utils/ptree";

async function supportsLinuxMountNamespaces(): Promise<boolean> {
	if (process.platform !== "linux") return false;
	try {
		const probe = Bun.spawn(
			[
				"unshare",
				"--user",
				"--map-root-user",
				"--mount",
				"--propagation",
				"private",
				"/bin/sh",
				"-c",
				"mount -t tmpfs tmpfs /proc",
			],
			{
				stdin: "ignore",
				stdout: "ignore",
				stderr: "ignore",
			},
		);
		return (await probe.exited) === 0;
	} catch {
		return false;
	}
}

const linuxMountNamespacesAvailable = await supportsLinuxMountNamespaces();

describe("ptree timeout", () => {
	it("contains the lifecycle rejection when the caller does not observe exited", async () => {
		const unhandled = new Set<unknown>();
		const onUnhandled = (reason: unknown) => {
			unhandled.add(reason);
		};
		process.on("unhandledRejection", onUnhandled);

		try {
			// Bun's subprocess timeout uses the platform clock; fake timers cannot drive this lifecycle.
			using child = spawn([process.execPath, "-e", "Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0)"], {
				timeout: 20,
			});
			await child.nothrow().text();
			await child.proc.exited;
			const nextTurn = Promise.withResolvers<void>();
			setImmediate(nextTurn.resolve);
			await nextTurn.promise;

			expect(child.exitReason).toBeInstanceOf(TimeoutError);
			expect(unhandled.has(child.exitReason)).toBe(false);
		} finally {
			process.off("unhandledRejection", onUnhandled);
		}
	});

	it.skipIf(process.platform !== "linux")(
		"kills descendants adopted while an AbortSignal races the timeout sweep",
		async () => {
			const testRoot = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-subreaper-race-"));
			const pidFile = path.join(testRoot, "worker.pid");
			const launcher = path.join(testRoot, "launcher.sh");
			await Bun.write(
				launcher,
				`#!/bin/sh
setsid sleep 30 </dev/null >/dev/null 2>&1 &
printf '%s\n' "$!" > ${JSON.stringify(pidFile)}
sleep 30
`,
			);
			await fs.chmod(launcher, 0o755);

			const controller = new AbortController();
			using child = spawn([launcher], {
				signal: controller.signal,
				subreaper: true,
			});
			const cleanupProcesses: Process[] = [];

			try {
				// A BunFile handle caches a negative `exists()`; stat fresh each poll.
				const pidFileExists = () =>
					fs.stat(pidFile).then(
						() => true,
						() => false,
					);
				const setupDeadline = Date.now() + 2_000;
				while (!(await pidFileExists()) && Date.now() < setupDeadline) await Bun.sleep(10);
				expect(await pidFileExists(), "the launcher must create its worker").toBe(true);

				const workerPid = Number.parseInt((await Bun.file(pidFile).text()).trim(), 10);
				const subreaper = Process.fromPid(child.pid);
				const command = subreaper?.children()[0];
				const worker = Process.fromPid(workerPid);
				if (!subreaper || !command || !worker) throw new Error("failed to capture the subreaper process tree");
				cleanupProcesses.push(worker, command, subreaper);
				expect(worker.ppid, "the worker must initially belong to the supervised command").toBe(command.pid);

				const killOnly = (pid: number): number => {
					try {
						process.kill(pid, "SIGKILL");
						return 1;
					} catch {
						return 0;
					}
				};
				const commandSnapshot = {
					killTree: () => killOnly(command.pid),
				} as unknown as Process;
				const pendingAdoption = {
					killTree: () => 0,
				} as unknown as Process;
				let snapshots = 0;
				let observedAdoption = false;
				const controlledSubreaper = {
					children: (): Process[] => {
						snapshots++;
						if (snapshots === 1) return [commandSnapshot];
						if (subreaper.status() !== ProcessStatus.Running || worker.status() !== ProcessStatus.Running)
							return [];
						if (worker.ppid !== subreaper.pid) return [pendingAdoption];
						observedAdoption = true;
						return [worker];
					},
					killTree: () => killOnly(subreaper.pid),
					terminate: () => Promise.resolve(killOnly(subreaper.pid) > 0),
				} as unknown as Process;
				const nativeFromPid = Process.fromPid.bind(Process);
				const fromPid = spyOn(Process, "fromPid").mockImplementation(pid =>
					pid === child.pid ? controlledSubreaper : nativeFromPid(pid),
				);

				try {
					child.kill(new TimeoutError(1, ""), -1);
					controller.abort("concurrent abort");

					const result = await child.wait({ allowAbort: true });
					expect(result.exitError).toBeInstanceOf(TimeoutError);
					expect(observedAdoption, "the worker must reparent to the live subreaper during cleanup").toBe(true);
					expect(worker.status(), `adopted descendant ${worker.pid} survived cleanup`).not.toBe(
						ProcessStatus.Running,
					);
				} finally {
					fromPid.mockRestore();
				}
			} finally {
				for (const processHandle of cleanupProcesses) processHandle.killTree(9);
				await fs.rm(testRoot, { recursive: true, force: true });
			}
		},
	);

	it.skipIf(process.platform !== "linux")("falls back after the first libc soname is unavailable", async () => {
		const script = createLinuxSubreaperScript(["libc.so.omp-missing", "libc.so.6", "libc.so"]);
		const child = Bun.spawn([process.execPath, "-e", script], {
			env: {
				...Bun.env,
				BUN_BE_BUN: "1",
				OMP_PTREE_SUBREAPER_COMMAND: JSON.stringify([
					process.execPath,
					"-e",
					'process.stdout.write("libc-fallback-ok")',
				]),
			},
			stdin: "ignore",
			stdout: "pipe",
			stderr: "pipe",
		});
		const [exitCode, stdout, stderr] = await Promise.all([
			child.exited,
			new Response(child.stdout).text(),
			new Response(child.stderr).text(),
		]);

		expect(exitCode, stderr).toBe(0);
		expect(stdout).toBe("libc-fallback-ok");
	});

	it.skipIf(process.platform !== "linux")("does not leak supervisor-only environment into the command", async () => {
		const result = await exec(["/bin/sh", "-c", `printf %s "\${BUN_BE_BUN-unset}"`], {
			subreaper: true,
		});

		expect(result.stdout).toBe("unset");
	});

	it.skipIf(process.platform !== "linux")("preserves caller-supplied BUN_BE_BUN for the command", async () => {
		const result = await exec(["/bin/sh", "-c", `printf %s "\${BUN_BE_BUN-unset}"`], {
			subreaper: true,
			env: { ...Bun.env, BUN_BE_BUN: "1" },
		});

		expect(result.stdout).toBe("1");
	});

	it.skipIf(!linuxMountNamespacesAvailable)("supervises commands without a mounted procfs", async () => {
		const script = `
const mountExit = await Bun.spawn(["mount", "-t", "tmpfs", "tmpfs", "/proc"], {
	stdout: "ignore",
	stderr: "inherit",
}).exited;
if (mountExit !== 0) throw new Error("failed to hide procfs");
${createLinuxSubreaperScript()}
`;
		const child = Bun.spawn(
			[
				"unshare",
				"--user",
				"--map-root-user",
				"--mount",
				"--propagation",
				"private",
				process.execPath,
				"-e",
				script,
			],
			{
				cwd: "/tmp",
				env: {
					...Bun.env,
					BUN_BE_BUN: "1",
					OMP_PTREE_SUBREAPER_COMMAND: JSON.stringify(["/bin/sh", "-c", "printf procfs-free-ok"]),
				},
				stdin: "ignore",
				stdout: "pipe",
				stderr: "pipe",
			},
		);
		const [exitCode, stdout, stderr] = await Promise.all([
			child.exited,
			new Response(child.stdout).text(),
			new Response(child.stderr).text(),
		]);

		expect(exitCode, stderr).toBe(0);
		expect(stdout).toBe("procfs-free-ok");
	});

	it("clears the timeout timer once the child exits so a fast command does not hold the event loop", async () => {
		// Real subprocess timing: the probe (a static-import fixture) resolves a
		// quick command under a 10 s ptree timeout and then must exit on its own;
		// if the timeout timer were left pending it would hold the probe's event
		// loop for the full 10 s.
		const probe = `${import.meta.dir}/fixtures/ptree-timeout-probe.ts`;

		const start = performance.now();
		const child = spawn([process.execPath, probe], { timeout: 15_000 });
		const text = await child.text();
		const elapsedMs = performance.now() - start;

		expect(text).toContain("probe-done");
		expect(elapsedMs).toBeLessThan(5_000);
	});

	it.skipIf(process.platform === "win32")(
		"keeps reading inherited stdout until the configured command deadline",
		async () => {
			// Real subprocess timing: fake timers cannot advance the child clock.
			// The root exits immediately, but its child writes after the legacy
			// 100 ms drain grace and before the 1 s command deadline.
			const result = await exec(["/bin/sh", "-c", "(sleep .2; printf token) &"], {
				timeout: 1_000,
				allowNonZero: true,
				allowAbort: true,
			});

			expect(result.ok).toBe(true);
			expect(result.stdout).toBe("token");
		},
	);

	it.skipIf(process.platform === "win32")("rejects text when the deadline fires after the root exits", async () => {
		using child = spawn(["/bin/sh", "-c", "sleep 30 & echo token"], {
			detached: true,
			timeout: 250,
		});
		let threw: unknown;
		try {
			await child.text();
		} catch (error) {
			threw = error;
		}

		expect(threw).toBeInstanceOf(TimeoutError);
	});

	for (const outputMethod of ["blob", "json", "arrayBuffer", "bytes"] as const) {
		it.skipIf(process.platform === "win32")(
			`rejects ${outputMethod} when the deadline fires after the root exits`,
			async () => {
				using child = spawn(["/bin/sh", "-c", `sleep 30 2>/dev/null & printf '"token"'`], {
					detached: true,
					timeout: 250,
				});
				let threw: unknown;
				try {
					await child[outputMethod]();
				} catch (error) {
					threw = error;
				}

				expect(threw).toBeInstanceOf(TimeoutError);
			},
		);
	}

	it.skipIf(process.platform === "win32")("keeps reading inherited stdout until EOF without a timeout", async () => {
		const result = await exec(["/bin/sh", "-c", "(sleep .2; printf token) &"], {
			allowNonZero: true,
			allowAbort: true,
		});

		expect(result.ok).toBe(true);
		expect(result.stdout).toBe("token");
	});

	it.skipIf(process.platform === "win32")(
		"terminates a detached group that holds stdout past the command deadline",
		async () => {
			// The root exits after printing its child's pid. The child keeps the
			// group and stdout alive past the deadline, so timeout must terminate
			// the group even though its original leader is already gone.
			let orphanPid: number | undefined;
			try {
				const result = await exec(["/bin/sh", "-c", "sleep 30 & echo $!"], {
					detached: true,
					timeout: 250,
					allowNonZero: true,
					allowAbort: true,
				});
				orphanPid = Number.parseInt(result.stdout.trim(), 10);

				expect(result.exitError).toBeInstanceOf(TimeoutError);
				// Real process state: SIGKILL delivery is synchronous, but pidfd
				// exit observation may settle on the next scheduler turn.
				const deadline = Date.now() + 500;
				let status = Process.fromPid(orphanPid)?.status();
				while (status === ProcessStatus.Running && Date.now() < deadline) {
					await Bun.sleep(10);
					status = Process.fromPid(orphanPid)?.status();
				}
				expect(status).not.toBe(ProcessStatus.Running);
			} finally {
				if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
			}
		},
	);

	it.skipIf(process.platform !== "win32")(
		"terminates a pipe-holding descendant after the Windows root exits",
		async () => {
			// Windows has no process groups. The probe exits after starting a
			// child that inherits stdout, so the retained root handle must anchor
			// the Toolhelp tree walk when the command deadline expires.
			const probe = `${import.meta.dir}/fixtures/ptree-dead-root-probe.ts`;
			let descendantPid: number | undefined;
			try {
				const result = await exec([process.execPath, probe], {
					timeout: 250,
					allowNonZero: true,
					allowAbort: true,
				});
				descendantPid = Number.parseInt(result.stdout.trim(), 10);

				expect(result.exitError).toBeInstanceOf(TimeoutError);
				const deadline = Date.now() + 500;
				let status = Process.fromPid(descendantPid)?.status();
				while (status === ProcessStatus.Running && Date.now() < deadline) {
					await Bun.sleep(10);
					status = Process.fromPid(descendantPid)?.status();
				}
				expect(status).not.toBe(ProcessStatus.Running);
			} finally {
				if (descendantPid) Process.fromPid(descendantPid)?.killTree(9);
			}
		},
	);

	it.skipIf(process.platform === "win32")(
		"throws NonZeroExitError by default when the child exits nonzero",
		async () => {
			// wait()'s default contract: without allowNonZero, a nonzero exit rejects
			// instead of returning an unsuccessful result.
			let threw: unknown;
			try {
				await exec(["sh", "-c", "exit 3"]);
			} catch (err) {
				threw = err;
			}
			expect(threw).toBeInstanceOf(NonZeroExitError);
		},
	);

	it.skipIf(process.platform === "win32")("completes when an orphan holds stdout past the root's exit", async () => {
		// `sleep 30 & echo token $!`: the root exits at once but the background
		// sleep inherits the pipe, so an EOF-based read would stall for the
		// orphan's lifetime, far past the timeout budget. The orphan's pid is
		// printed so the fixture can clean it up instead of leaking it.
		let orphanPid: number | undefined;
		try {
			const start = performance.now();
			const result = await exec(["sh", "-c", "sleep 30 & echo token $!"], {
				timeout: 1_000,
				allowNonZero: true,
				allowAbort: true,
			});
			const elapsedMs = performance.now() - start;
			const match = /^token (\d+)$/.exec(result.stdout.trim());
			orphanPid = match ? Number.parseInt(match[1], 10) : undefined;
			expect(result.ok).toBe(true);
			expect(match, `stdout was: ${result.stdout}`).not.toBeUndefined();
			expect(elapsedMs).toBeLessThan(5_000);
		} finally {
			if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
		}
	});

	it.skipIf(process.platform === "win32")(
		"applies the deadline when the root exited before the timeout was attached",
		async () => {
			// Under load Bun can reap the root before spawn() returns, so
			// attachTimeout() sees `proc.killed`; the orphan still holds stdout and
			// only the deadline's cutoff ends the read. Real subprocess timing: the
			// cutoff races a live pipe, which fake timers cannot drive.
			const proc = Bun.spawn(["sh", "-c", "sleep 30 & echo token $!"], {
				stdin: "ignore",
				stdout: "pipe",
				stderr: "pipe",
			});
			await proc.exited;
			expect(proc.killed).toBe(true);
			let orphanPid: number | undefined;
			try {
				const child = new ChildProcess(proc, false);
				child.attachTimeout(250);
				const text = await child.nothrow().text();
				const match = /^token (\d+)$/.exec(text.trim());
				orphanPid = match ? Number.parseInt(match[1], 10) : undefined;
				expect(match, `stdout was: ${text}`).not.toBeNull();
			} finally {
				if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
			}
		},
	);

	it.skipIf(process.platform === "win32")("completes when an orphan holds stderr past the root's exit", async () => {
		let orphanPid: number | undefined;
		try {
			const start = performance.now();
			const result = await exec(["sh", "-c", "sleep 30 >&2 & echo token2 $!"], {
				timeout: 1_000,
				allowNonZero: true,
				allowAbort: true,
			});
			const elapsedMs = performance.now() - start;
			const match = /^token2 (\d+)$/.exec(result.stdout.trim());
			orphanPid = match ? Number.parseInt(match[1], 10) : undefined;
			expect(result.ok).toBe(true);
			expect(match, `stdout was: ${result.stdout}`).not.toBeUndefined();
			expect(elapsedMs).toBeLessThan(5_000);
		} finally {
			if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
		}
	});

	it.skipIf(process.platform === "win32")("completes when a nonzero exit races an orphan holding stderr", async () => {
		// `sleep 30 >&2 & exit 1`: the nonzero-exit normalization awaits the
		// stderr drain, so a grace keyed on the normalized exit promise would
		// deadlock until the orphan closes stderr. The grace must key on the
		// raw process exit.
		let orphanPid: number | undefined;
		try {
			const start = performance.now();
			const result = await exec(["sh", "-c", "sleep 30 >&2 & echo $! >&2; exit 1"], {
				timeout: 1_000,
				allowNonZero: true,
				allowAbort: true,
			});
			const elapsedMs = performance.now() - start;
			const match = /(\d+)\s*$/.exec(result.stderr.trim());
			orphanPid = match ? Number.parseInt(match[1], 10) : undefined;
			expect(result.exitCode).toBe(1);
			expect(elapsedMs).toBeLessThan(5_000);
		} finally {
			if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
		}
	});

	it.skipIf(process.platform === "win32")(
		"preserves the timeout reason when nonzero normalization waits for stderr",
		async () => {
			// The root exits nonzero while its child holds stderr open. The
			// deadline kills the detached group while exit normalization awaits
			// the drain, and that timeout must outrank the earlier exit code.
			let threw: unknown;
			try {
				await exec(["/bin/sh", "-c", "sleep 30 >&2 & exit 7"], {
					detached: true,
					timeout: 250,
					allowNonZero: true,
				});
			} catch (err) {
				threw = err;
			}

			expect(threw).toBeInstanceOf(TimeoutError);
		},
	);

	it.skipIf(process.platform === "win32")(
		"bounds post-exit drain across stdout and stderr on detached non-subreaper spawns while preserving real exit code and killing helpers",
		async () => {
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-drain-nonsub-"));
			try {
				// 1. Helper holds ONLY stderr open on exit 23
				const stderrSentinel = path.join(dir, "stderr-helper.txt");
				const t1 = Date.now();
				const cp1 = spawn(
					[
						"/bin/sh",
						"-c",
						`(trap '' TERM; sleep 0.5 && echo lived > ${JSON.stringify(stderrSentinel)}) >/dev/null & echo err-msg >&2; exit 23`,
					],
					{ detached: true, subreaper: false, descendantDrainMs: 120 },
				);
				expect(await cp1.exited).toBe(23);
				expect(cp1.peekStderr()).toContain("err-msg");
				expect(Date.now() - t1).toBeLessThan(450);
				await Bun.sleep(550);
				expect(await Bun.file(stderrSentinel).exists()).toBe(false);

				// 2. wait() with helper holding ONLY stdout open (stderr closes immediately)
				const stdoutSentinel = path.join(dir, "stdout-helper.txt");
				const t2 = Date.now();
				const cp2 = spawn(
					[
						"/bin/sh",
						"-c",
						`(trap '' TERM; sleep 0.5 && echo lived > ${JSON.stringify(stdoutSentinel)}) 2>/dev/null & echo out-ok; exit 0`,
					],
					{ detached: true, subreaper: false, descendantDrainMs: 120 },
				);
				const res2 = await cp2.wait({ allowNonZero: true });
				expect(res2.exitCode).toBe(0);
				expect(res2.stdout).toContain("out-ok");
				expect(Date.now() - t2).toBeLessThan(450);
				await Bun.sleep(550);
				expect(await Bun.file(stdoutSentinel).exists()).toBe(false);

				// 3. wait() with stderr closing 40ms into the drain while stdout stays open
				const mixedSentinel = path.join(dir, "mixed-helper.txt");
				const t3 = Date.now();
				const cp3 = spawn(
					[
						"/bin/sh",
						"-c",
						`(trap '' TERM; sleep 0.04) >/dev/null & (trap '' TERM; sleep 0.5 && echo lived > ${JSON.stringify(mixedSentinel)}) 2>/dev/null & echo mixed-ok; exit 0`,
					],
					{ detached: true, subreaper: false, descendantDrainMs: 120 },
				);
				const res3 = await cp3.wait({ allowNonZero: true });
				expect(res3.exitCode).toBe(0);
				expect(res3.stdout).toContain("mixed-ok");
				expect(Date.now() - t3).toBeLessThan(450);
				await Bun.sleep(550);
				expect(await Bun.file(mixedSentinel).exists()).toBe(false);

				// 4. Streaming readStdout() with exit 23 and stdout-only helper
				const streamSentinel = path.join(dir, "stream-helper.txt");
				const t4 = Date.now();
				const cp4 = spawn(
					[
						"/bin/sh",
						"-c",
						`(trap '' TERM; sleep 0.5 && echo lived > ${JSON.stringify(streamSentinel)}) 2>/dev/null & echo stream-ok; exit 23`,
					],
					{ detached: true, subreaper: false, descendantDrainMs: 120 },
				);
				let streamed = "";
				const dec = new TextDecoder();
				const readP = cp4.readStdout(chunk => {
					streamed += dec.decode(chunk, { stream: true });
				});
				expect(await cp4.exited).toBe(23);
				await Promise.all([readP, cp4.stderrDone]);
				expect(streamed).toContain("stream-ok");
				expect(Date.now() - t4).toBeLessThan(450);
				await Bun.sleep(550);
				expect(await Bun.file(streamSentinel).exists()).toBe(false);

				// 5. Late readStdout() registered after the post-exit drain deadline preserves ready stdout, expires immediately, and kills helper
				const lateSentinel = path.join(dir, "late-helper.txt");
				const cp5 = spawn(
					[
						"/bin/sh",
						"-c",
						`(trap '' TERM; sleep 0.6 && echo lived > ${JSON.stringify(lateSentinel)}) 2>/dev/null & printf late-ok; exit 0`,
					],
					{ detached: true, subreaper: false, descendantDrainMs: 60 },
				);
				expect(await cp5.exited).toBe(0);
				await Bun.sleep(140);
				const tLate = performance.now();
				let lateOut = "";
				const lateDec = new TextDecoder();
				await cp5.readStdout(chunk => {
					lateOut += lateDec.decode(chunk, { stream: true });
				});
				lateOut += lateDec.decode();
				expect(lateOut).toBe("late-ok");
				expect(performance.now() - tLate).toBeLessThan(60);
				await Bun.sleep(550);
				expect(await Bun.file(lateSentinel).exists()).toBe(false);

				// 6. Late readStdout(), wait(), and bytes() after deadline with NO descendants preserve ready stdout
				for (const mode of ["readStdout", "wait", "bytes"] as const) {
					const cpNoHelper = spawn(["/bin/sh", "-c", "printf normal-output; exit 0"], {
						detached: true,
						subreaper: false,
						descendantDrainMs: 40,
					});
					expect(await cpNoHelper.exited).toBe(0);
					await Bun.sleep(100);
					if (mode === "readStdout") {
						let out = "";
						const d = new TextDecoder();
						await cpNoHelper.readStdout(c => {
							out += d.decode(c, { stream: true });
						});
						out += d.decode();
						expect(out).toBe("normal-output");
					} else if (mode === "wait") {
						const r = await cpNoHelper.wait();
						expect(r.stdout).toBe("normal-output");
					} else {
						const b = await cpNoHelper.bytes();
						expect(new TextDecoder().decode(b)).toBe("normal-output");
					}
				}

				// 7. Backward wall-clock jump (Date.now) does not extend the monotonic post-exit drain deadline.
				//    Use detached:false with a finite-lived inherited stdout holder (portable across macOS/Linux without setsid)
				//    so the holder remains alive when the late reader starts.
				const holderPidFile = path.join(dir, "clock-holder.pid");
				const cpClock = spawn(
					[
						"/bin/sh",
						"-c",
						`(sh -c 'echo $$ > ${JSON.stringify(holderPidFile)}; sleep 0.6' </dev/null 2>/dev/null) & printf clock-ok; exit 0`,
					],
					{ detached: false, subreaper: false, descendantDrainMs: 50 },
				);
				expect(await cpClock.exited).toBe(0);
				await Bun.sleep(130);
				const holderPid = Number.parseInt(
					(
						await Bun.file(holderPidFile)
							.text()
							.catch(() => "")
					).trim(),
					10,
				);
				expect(holderPid).toBeGreaterThan(1);
				// Confirm the stdout holder is still alive when the late reader starts
				expect(() => process.kill(holderPid, 0)).not.toThrow();
				const realDateNow = Date.now;
				const dateNowSpy = spyOn(Date, "now").mockImplementation(() => realDateNow() - 10_000);
				try {
					const tClock = performance.now();
					let clockOut = "";
					const d = new TextDecoder();
					await cpClock.readStdout(c => {
						clockOut += d.decode(c, { stream: true });
					});
					clockOut += d.decode();
					expect(clockOut).toBe("clock-ok");
					expect(performance.now() - tClock).toBeLessThan(60);
				} finally {
					dateNowSpy.mockRestore();
					try {
						process.kill(holderPid, "SIGKILL");
					} catch {}
				}

				// 8. Standalone one-shot caller awaiting wait() kills closed-pipe same-group helpers before exiting
				const closedPipeSentinel = path.join(dir, "closed-pipe-helper.txt");
				const oneShotScript = path.join(dir, "one-shot-wait.ts");
				await Bun.write(
					oneShotScript,
					`import { spawn } from ${JSON.stringify(path.join(import.meta.dir, "../src/ptree.ts"))};
const cp = spawn(
	["/bin/sh", "-c", ${JSON.stringify(`(trap '' TERM; sleep 0.5 && echo lived > ${JSON.stringify(closedPipeSentinel)}) </dev/null >/dev/null 2>&1 & printf root-ok; exit 0`)}],
	{ detached: true, subreaper: false, descendantDrainMs: 120 },
);
const res = await cp.wait();
process.stdout.write(res.stdout + ":" + res.exitCode);
`,
				);
				const caller = Bun.spawn([process.execPath, oneShotScript], { stdout: "pipe", stderr: "pipe" });
				const [callerExit, callerOut] = await Promise.all([caller.exited, new Response(caller.stdout).text()]);
				expect(callerExit).toBe(0);
				expect(callerOut).toBe("root-ok:0");
				await Bun.sleep(550);
				expect(await Bun.file(closedPipeSentinel).exists()).toBe(false);

				// 9. Closed-pipe descendant settlement honors both command timeout and AbortSignal
				for (const kind of ["timeout", "abort"] as const) {
					const cutSentinel = path.join(dir, `closed-pipe-${kind}.txt`);
					const ac = new AbortController();
					const tCut = performance.now();
					const cpCut = spawn(
						[
							"/bin/sh",
							"-c",
							`(trap '' TERM; sleep 0.4 && echo lived > ${JSON.stringify(cutSentinel)}) </dev/null >/dev/null 2>&1 & printf cut-ok; exit 0`,
						],
						{
							detached: true,
							subreaper: false,
							descendantDrainMs: 750,
							...(kind === "timeout" ? { timeout: 150 } : { signal: ac.signal }),
						},
					);
					const abortTimer = kind === "abort" ? setTimeout(() => ac.abort("stop-drain"), 150) : undefined;
					let caught: unknown;
					try {
						await cpCut.wait();
					} catch (err) {
						caught = err;
					} finally {
						if (abortTimer) clearTimeout(abortTimer);
					}
					expect(performance.now() - tCut).toBeLessThan(500);
					if (kind === "timeout") {
						expect(caught).toBeInstanceOf(TimeoutError);
					} else {
						expect(caught).toBeInstanceOf(AbortError);
					}
					await Bun.sleep(450);
					expect(await Bun.file(cutSentinel).exists()).toBe(false);
				}

				// 10. A late abort after drain settlement (natural exit, closed-stdio expiry, and pipe-holding expiry) must not fail a completed wait()
				for (const mode of ["natural", "expired", "expired-piped"] as const) {
					const acLate = new AbortController();
					const helperSleep = mode === "natural" ? "0.04" : "0.5";
					const drainBudget = mode === "natural" ? 300 : 60;
					const cmd =
						mode === "expired-piped"
							? `(trap '' TERM; sleep ${helperSleep}) </dev/null & printf finished; exit 0`
							: `(trap '' TERM; sleep ${helperSleep}) </dev/null >/dev/null 2>&1 & printf finished; exit 0`;
					const cpLateAbort = spawn(["/bin/sh", "-c", cmd], {
						detached: true,
						subreaper: false,
						descendantDrainMs: drainBudget,
						signal: acLate.signal,
					});
					const waitP = cpLateAbort.wait({ allowAbort: true });
					void cpLateAbort.exited.then(() =>
						cpLateAbort.drainDone.then(() => {
							acLate.abort("too-late");
						}),
					);
					const resLate = await waitP;
					expect(resLate.stdout).toBe("finished");
					expect(resLate.exitCode).toBe(0);
					expect(resLate.ok).toBe(true);
					expect(resLate.exitError).toBeUndefined();
					expect(cpLateAbort.exitReason).toBeUndefined();
				}

				// 11. Aborting closed-stdio drain after a signal-terminated leader kills descendants and reports AbortError
				{
					const sigSentinel = path.join(dir, "signal-drain-abort.txt");
					const acSig = new AbortController();
					const cpSig = spawn(
						[
							"/bin/sh",
							"-c",
							`(trap '' TERM; sleep 0.25; echo survived > "$1") </dev/null >/dev/null 2>&1 & kill -TERM $$`,
							"_",
							sigSentinel,
						],
						{ detached: true, subreaper: false, descendantDrainMs: 2500, signal: acSig.signal },
					);
					const tSig = performance.now();
					let sigAbortTimer: ReturnType<typeof setTimeout> | undefined;
					void cpSig.exited
						.catch(() => {})
						.then(() => {
							sigAbortTimer = setTimeout(() => acSig.abort("cancel-after-signal"), 30);
						});
					let caught: unknown;
					try {
						await cpSig.wait();
					} catch (err) {
						caught = err;
					} finally {
						if (sigAbortTimer) clearTimeout(sigAbortTimer);
					}
					expect(performance.now() - tSig).toBeLessThan(600);
					expect(caught).toBeInstanceOf(AbortError);
					await Bun.sleep(350);
					expect(await Bun.file(sigSentinel).exists()).toBe(false);
				}
			} finally {
				await fs.rm(dir, { recursive: true, force: true });
			}
		},
		30_000,
	);
});

describe.skipIf(process.platform !== "linux")("ptree subreaper stop", () => {
	async function readPid(pidFile: string): Promise<number> {
		// Polls a file a real process writes; returns as soon as it holds a pid.
		const deadline = Date.now() + 10_000;
		while (Date.now() < deadline) {
			const pid = Number.parseInt(
				await Bun.file(pidFile)
					.text()
					.catch(() => ""),
				10,
			);
			if (pid > 0) return pid;
			await Bun.sleep(10);
		}
		throw new Error("command never wrote its pid");
	}

	async function runningAfter(pid: number, ms: number): Promise<boolean> {
		const deadline = Date.now() + ms;
		let status = Process.fromPid(pid)?.status();
		while (status === ProcessStatus.Running && Date.now() < deadline) {
			await Bun.sleep(10);
			status = Process.fromPid(pid)?.status();
		}
		return status === ProcessStatus.Running;
	}

	const stops = [
		["an AbortSignal", (child: ChildProcess, controller: AbortController) => controller.abort()],
		["dispose", (child: ChildProcess) => child[Symbol.dispose]()],
		["a graceful kill", (child: ChildProcess) => child.kill(new AbortError(new Error("stop"), ""), 500)],
		["a hard kill", (child: ChildProcess) => child.kill(new AbortError(new Error("stop"), ""), -1)],
	] as const;

	for (const [name, stop] of stops) {
		it(`ends a descendant that ignores SIGTERM on ${name}`, async () => {
			// The shell and its background child both ignore SIGTERM (an ignored
			// disposition survives exec). Only the stop's SIGKILL can end them,
			// and only while the subreaper still holds the orphaned child.
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-subreaper-stop-"));
			const pidFile = path.join(dir, "pid");
			let orphanPid: number | undefined;
			try {
				const controller = new AbortController();
				const child = spawn(["/bin/sh", "-c", `trap "" TERM; sleep 3600 & echo $! > '${pidFile}'; wait`], {
					subreaper: true,
					signal: controller.signal,
				});
				const outcome = child.wait().then(
					() => new Error("run finished without the stop"),
					(error: unknown) => error,
				);
				orphanPid = await readPid(pidFile);
				const stoppedAt = Date.now();

				stop(child, controller);

				expect(await outcome).toBeInstanceOf(AbortError);
				expect(await runningAfter(orphanPid, 2_000)).toBe(false);
				expect(Date.now() - stoppedAt).toBeLessThan(10_000);
			} finally {
				if (orphanPid) Process.fromPid(orphanPid)?.killTree(9);
				await fs.rm(dir, { recursive: true, force: true });
			}
		}, 30_000);
	}

	// Only a stop with a grace period can promise the handler runs: a zero grace
	// sends SIGKILL right after SIGTERM, racing the trap.
	for (const [name, stop] of [
		["an AbortSignal", (_child: ChildProcess, controller: AbortController) => controller.abort()],
	] as const) {
		it(`gives the command its SIGTERM before the hard kill on ${name}`, async () => {
			// A command that cleans up on SIGTERM must get the chance on a graceful
			// stop, not only a SIGKILL.
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-subreaper-stop-"));
			const pidFile = path.join(dir, "pid");
			const marker = path.join(dir, "cleaned");
			try {
				const controller = new AbortController();
				const child = spawn(
					[
						"/bin/sh",
						"-c",
						`trap 'echo cleaned > "${marker}"; exit 0' TERM; echo $$ > '${pidFile}'; sleep 3600 & wait`,
					],
					{ subreaper: true, signal: controller.signal },
				);
				const outcome = child.wait().catch((error: unknown) => error);
				await readPid(pidFile);

				stop(child, controller);

				expect(await outcome).toBeInstanceOf(AbortError);
				expect(await Bun.file(marker).text()).toBe("cleaned\n");
			} finally {
				await fs.rm(dir, { recursive: true, force: true });
			}
		}, 30_000);
	}

	for (const [name, start] of [
		[
			"its signal is already aborted",
			(cmd: string[], env: Record<string, string | undefined>) =>
				spawn(cmd, { subreaper: true, env, signal: AbortSignal.abort() }),
		],
		[
			"it is killed before the subreaper launches it",
			(cmd: string[], env: Record<string, string | undefined>) => {
				const child = spawn(cmd, { subreaper: true, env });
				child.kill(new AbortError(new Error("stop"), ""));
				return child;
			},
		],
	] as const) {
		it(`never runs the command when ${name}`, async () => {
			// The subreaper has no child yet, so there is nothing to warn: the
			// stop must end the subreaper before it launches the cancelled command,
			// not wait out a grace period during which the command starts.
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-subreaper-stop-"));
			const marker = path.join(dir, "ran");
			// A loaded runner can stall this process between spawn() and the stop
			// for longer than the subreaper takes to boot and launch the command.
			// A preload holds the subreaper's Bun until the stop has been issued, so
			// the command can only start if the stop failed to end the subreaper.
			const go = path.join(dir, "go");
			const gate = path.join(dir, "gate.ts");
			await Bun.write(
				gate,
				`import { existsSync } from "node:fs";\nconst until = Date.now() + 10_000;\nwhile (!existsSync(${JSON.stringify(go)}) && Date.now() < until) Bun.sleepSync(5);\n`,
			);
			try {
				const child = start(["/bin/sh", "-c", `echo ran > '${marker}'; sleep 5`], {
					...Bun.env,
					BUN_OPTIONS: `--preload=${gate}`,
				});
				await Bun.write(go, "");
				expect(await child.wait().catch((error: unknown) => error)).toBeInstanceOf(AbortError);
				// A command the subreaper launched in its last instant would be a real
				// process outside our control; no event marks "it never wrote", so give
				// such an escaped command real time to write before checking.
				await Bun.sleep(100);
				expect(await fs.exists(marker)).toBe(false);
			} finally {
				await fs.rm(dir, { recursive: true, force: true });
			}
		}, 30_000);
	}

	for (const [name, escalate] of [
		["the command exits on SIGTERM", false],
		["a hard kill ends the grace period", true],
	] as const) {
		it(`lets the caller's process exit once the stop is over when ${name}`, async () => {
			// The stop runs in a fresh Bun process with a 10 s grace period. Once
			// wait() has settled, nothing of the stop may keep that process alive.
			// For the hard kill, the command ignores SIGTERM, so only the hard kill
			// ends the grace period.
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-subreaper-stop-"));
			const pidFile = path.join(dir, "pid");
			const script = path.join(dir, "stop.ts");
			const onTerm = escalate ? `trap "" TERM` : `trap 'exit 0' TERM`;
			await Bun.write(
				script,
				`import { AbortError, spawn } from ${JSON.stringify(path.join(import.meta.dir, "../src/ptree.ts"))};
const child = spawn(["/bin/sh", "-c", ${JSON.stringify(`${onTerm}; echo $$ > '${pidFile}'; sleep 3600 & wait`)}], { subreaper: true });
const outcome = child.wait().catch(() => {});
while (!(await Bun.file(${JSON.stringify(pidFile)}).text().catch(() => ""))) await Bun.sleep(10);
child.kill(new AbortError(new Error("stop"), ""), 10_000);
${escalate ? `await Bun.sleep(50); child.kill(new AbortError(new Error("stop"), ""), -1);` : ""}
await outcome;
console.log("settled");
`,
			);
			try {
				const caller = Bun.spawn([process.execPath, script], { stdout: "pipe", stderr: "inherit" });
				const reader = caller.stdout.getReader();
				let output = "";
				while (!output.includes("settled")) {
					const chunk = await reader.read();
					if (chunk.done) break;
					output += new TextDecoder().decode(chunk.value);
				}
				const settledAt = Date.now();
				reader.releaseLock();

				expect(output).toContain("settled");
				const exitCode = await caller.exited;
				expect(exitCode).toBe(0);
				expect(Date.now() - settledAt).toBeLessThan(3_000);
			} finally {
				await fs.rm(dir, { recursive: true, force: true });
			}
		}, 60_000);
	}

	it.skipIf(process.platform !== "linux")(
		"kills pgrp helpers and fails closed on surviving setsid helpers when procfs discovery fails",
		async () => {
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-utils-drain-noproc-"));
			const pgrpSentinel = path.join(dir, "pgrp-lived.txt");
			const setsidPidFile = path.join(dir, "setsid.pid");
			const brokenProcScript = createLinuxSubreaperScript().replaceAll('"/proc/"', '"/nonexistent-proc-dir/"');
			try {
				// Non-setsid helper is still killed via kill(-child.pid, 9) even when /proc fails
				const pgrpProc = Bun.spawn([process.execPath, "-e", brokenProcScript], {
					env: {
						...Bun.env,
						BUN_BE_BUN: "1",
						OMP_PTREE_SUBREAPER_COMMAND: JSON.stringify([
							"/bin/sh",
							"-c",
							`(trap '' TERM; sleep 0.6 && echo lived > ${JSON.stringify(pgrpSentinel)}) >/dev/null 2>&1 & exit 0`,
						]),
						OMP_PTREE_SUBREAPER_DRAIN_MS: "80",
					},
					stdin: "ignore",
					stdout: "pipe",
					stderr: "pipe",
				});
				expect(await pgrpProc.exited).toBe(0);
				await Bun.sleep(750);
				expect(await Bun.file(pgrpSentinel).exists()).toBe(false);

				// Session-detached helper that cannot be discovered without /proc forces nonzero exit (fail closed)
				const setsidProc = Bun.spawn([process.execPath, "-e", brokenProcScript], {
					env: {
						...Bun.env,
						BUN_BE_BUN: "1",
						OMP_PTREE_SUBREAPER_COMMAND: JSON.stringify([
							"/bin/sh",
							"-c",
							`setsid sh -c 'echo $$ > ${JSON.stringify(setsidPidFile)}; sleep 5' </dev/null >/dev/null 2>&1 & sleep 0.05; exit 0`,
						]),
						OMP_PTREE_SUBREAPER_DRAIN_MS: "80",
					},
					stdin: "ignore",
					stdout: "pipe",
					stderr: "pipe",
				});
				const [setsidExit, setsidStderr] = await Promise.all([
					setsidProc.exited,
					new Response(setsidProc.stderr).text(),
				]);
				expect(setsidExit).toBe(1);
				expect(setsidStderr).toContain("failed to terminate lingering descendants");
			} finally {
				const rawPid = await Bun.file(setsidPidFile)
					.text()
					.catch(() => "");
				const pid = Number.parseInt(rawPid.trim(), 10);
				if (pid > 1) {
					try {
						process.kill(pid, "SIGKILL");
					} catch {}
				}
				await fs.rm(dir, { recursive: true, force: true });
			}
		},
	);

	it.skipIf(process.platform === "win32")(
		"kills SIGINT-ignoring descendants in a non-subreaper detached group after the leader exits on SIGINT",
		async () => {
			const dir = await fs.mkdtemp(path.join(os.tmpdir(), "ptree-sigint-group-"));
			const helperScript = path.join(dir, "helper.sh");
			const helperPidFile = path.join(dir, "helper.pid");
			const leaderSigFile = path.join(dir, "leader.sig");
			const readyFile = path.join(dir, "ready");
			try {
				await fs.writeFile(
					helperScript,
					`trap '' INT TERM\necho $$ > ${JSON.stringify(helperPidFile)}\nexec sleep 15\n`,
					"utf8",
				);
				const leaderCmd = `trap 'echo SIGINT > ${JSON.stringify(leaderSigFile)}; exit 130' INT; trap 'echo SIGTERM > ${JSON.stringify(leaderSigFile)}; exit 143' TERM; /bin/sh ${JSON.stringify(helperScript)} </dev/null >/dev/null 2>&1 & while [ ! -s ${JSON.stringify(helperPidFile)} ]; do sleep 0.01; done; echo ready > ${JSON.stringify(readyFile)}; sleep 15`;
				const cp = spawn(["/bin/sh", "-c", leaderCmd], {
					detached: true,
					subreaper: false,
					descendantDrainMs: 100,
				});
				const deadline = Date.now() + 3000;
				while (!(await Bun.file(readyFile).exists()) && Date.now() < deadline) {
					await Bun.sleep(10);
				}
				expect(await Bun.file(readyFile).exists()).toBe(true);
				const helperPid = Number.parseInt((await Bun.file(helperPidFile).text()).trim(), 10);
				expect(helperPid).toBeGreaterThan(1);
				expect(helperPid).not.toBe(cp.pid);
				expect(processInfo([helperPid]).map(p => p.pid)).toEqual([helperPid]);

				cp.kill(new AbortError("steer", ""), 120, "SIGINT");
				await cp.wait({ allowAbort: true });

				expect((await Bun.file(leaderSigFile).text()).trim()).toBe("SIGINT");

				// The SIGINT/SIGTERM-ignoring descendant must be hard-killed by the group SIGKILL sweep.
				// Native `processInfo` excludes exited and zombie processes without swallowing probe errors.
				const stopDeadline = Date.now() + 1000;
				while (processInfo([helperPid]).length > 0 && Date.now() < stopDeadline) {
					await Bun.sleep(20);
				}
				expect(processInfo([helperPid])).toEqual([]);
			} finally {
				const rawPid = await Bun.file(helperPidFile)
					.text()
					.catch(() => "");
				const pid = Number.parseInt(rawPid.trim(), 10);
				if (pid > 1) {
					try {
						process.kill(pid, "SIGKILL");
					} catch {}
				}
				await fs.rm(dir, { recursive: true, force: true });
			}
		},
	);
});
