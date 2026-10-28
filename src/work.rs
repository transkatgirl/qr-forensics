//! Calling a slow step off, and spreading one over the cores that stand
//! idle.
//!
//! A search of a photo climbs a ladder of copies and fits a placement to
//! every outline the finder patterns propose; a read tries every kernel
//! under every threshold rule, and then the photo's other three turns. On
//! a sound symbol the first try settles it. On a damaged one, which is
//! what this crate is for, every try is made and fails, one after another
//! on one core, for minutes. Two things mend that.
//!
//! The tries a step would make in turn are made side by side, on the
//! cores nothing else is using, and taken in the order they would have
//! been made in, so that what a step finds is what it found on one core.
//! The first try is still made alone wherever it most often settles the
//! matter: made beside the rest, it would have to wait for them. The
//! tries go to threads started once and kept ([`pool`]), so that a step
//! of a few milliseconds is as well worth spreading as one of minutes.
//!
//! And whoever started a step may call it off: the loops that take the
//! time ask [`cancelled`] as they go round, and come back with what they
//! have. The request to stop is not handed down through every function
//! between the caller and those loops. It is set for the thread the work
//! runs on, by [`Cancel::scope`], and handed on to every thread the work
//! is spread over.

use std::any::Any;
use std::cell::RefCell;
use std::panic::{self, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, mpsc};

use rayon::{ThreadPool, ThreadPoolBuilder};

/// A request to stop, shared between whoever makes it and the work it is
/// made of.
#[derive(Clone, Debug, Default)]
pub struct Cancel {
    asked: Arc<AtomicBool>,
    /// The request the work was already answering to when this one was
    /// made for a part of it: the part stops for either.
    outer: Option<Arc<Cancel>>,
}

thread_local! {
    /// The request the work on this thread answers to.
    static CURRENT: RefCell<Option<Cancel>> = const { RefCell::new(None) };
}

/// Puts back the request a thread answered to before a scope.
struct Restore(Option<Cancel>);

impl Drop for Restore {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }

    /// A request for a part of the work this thread is doing, which
    /// stops when this is asked or when the whole is called off.
    fn within_current() -> Self {
        Self { asked: Arc::default(), outer: current().map(Arc::new) }
    }

    /// Ask the work to stop. It stops at the next turn of whichever loop
    /// it is in, not at once.
    pub fn cancel(&self) {
        self.asked.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.asked.load(Ordering::Relaxed) || self.outer.as_ref().is_some_and(|outer| outer.is_cancelled())
    }

    /// Run `work` answering to this request: [`cancelled`], asked on this
    /// thread or on any the work is spread over, says whether it has
    /// been made.
    pub fn scope<R>(&self, work: impl FnOnce() -> R) -> R {
        answering(Some(self), work)
    }
}

/// `work` run answering to `request`, or to none, whatever this thread
/// answered to before: a thread of the [`pool`] takes up the tasks of
/// one step after those of another, and each task answers to the
/// request of the step it is part of.
fn answering<R>(request: Option<&Cancel>, work: impl FnOnce() -> R) -> R {
    let _restore = Restore(CURRENT.with(|current| current.replace(request.cloned())));
    work()
}

fn current() -> Option<Cancel> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Whether the work this thread is doing has been called off. Work that
/// answers to no request is never called off.
pub fn cancelled() -> bool {
    CURRENT.with(|current| current.borrow().as_ref().is_some_and(Cancel::is_cancelled))
}

/// The threads a step is spread over, besides the caller's own: one for
/// every core but one, started with the first step spread and kept for
/// every step after it. A step hands its tries to them as tasks, where
/// it once started threads of its own at twenty microseconds of the
/// processor's time each, which kept the shorter steps, such as the
/// scoring of a window of placements, to one core. A thread works one
/// task at a time, and one whose task waits on the tasks of a step
/// inside it waits as the caller of any step does, taking up nothing
/// else meanwhile: a task taken up then, of another step, would hold
/// the step it waits in back for as long as that task took, however
/// soon its own were done.
fn pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
        ThreadPoolBuilder::new()
            .num_threads(cores.saturating_sub(1).max(1))
            .thread_name(|i| format!("work {i}"))
            .build()
            .expect("the threads of the work start")
    })
}

/// Threads of the [`pool`] at work on a task; the rest stand idle.
static AT_WORK: AtomicUsize = AtomicUsize::new(0);

/// Threads of the [`pool`] standing idle just now.
fn idle() -> usize {
    pool().current_num_threads().saturating_sub(AT_WORK.load(Ordering::Relaxed))
}

/// Up to so many of the [`pool`]'s threads for one step, besides the
/// caller's own: a bound on how many of the step's items are worked at
/// once, for work dear in memory, rather than a claim on cores. The
/// threads go to whichever task they find, so a step inside another asks
/// for as many as it could use, and those not working elsewhere take its
/// items up; those that never come to it cost it nothing, since it waits
/// only for the tasks that have begun ([`Helpers`]). Counting the cores
/// out over the whole process instead, as before there was a pool, a
/// step inside a step that held them waited on one core while threads of
/// the pool stood idle.
pub struct Crew {
    extra: usize,
}

impl Crew {
    /// Up to `most` of the pool's threads.
    pub fn gather(most: usize) -> Self {
        Self { extra: most.min(pool().current_num_threads()) }
    }

    /// Up to `most` of the pool's threads standing idle just now: for
    /// items worked ahead of need, which those before them may make
    /// worthless, and which are worth a thread only when it would
    /// otherwise do nothing. With every thread at work elsewhere, a crew
    /// of [`Self::gather`] leaves the caller to work them all itself, one
    /// after another, before it can take the first.
    pub fn gather_idle(most: usize) -> Self {
        Self { extra: most.min(idle()) }
    }

    /// How many items the crew works at once: the caller, and the pool's
    /// threads it may have.
    pub fn width(&self) -> usize {
        self.extra + 1
    }

    /// `work` done on every item, the results in the items' order
    /// whichever was finished first.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
        spread(self.width(), items, work)
    }
}

/// The tasks a step hands the [`pool`], as its caller keeps count of
/// them. What a task works on lies on the caller's stack, so the caller
/// may not return while one is at it; but a task not yet begun need not
/// be waited for, and with every thread of the pool busy with the long
/// tries of another step it would not begin until they ended, though
/// the caller had worked every item itself meanwhile. So a task begins
/// only while its step is open ([`Self::enter`]), and the caller, done
/// with the step, closes it and waits for the tasks under way alone
/// ([`Self::close`]): one that begins after that finds the step closed
/// and ends at once, touching nothing of the caller's.
struct Helpers {
    /// Whether the step is closed, and how many of its tasks are under
    /// way.
    state: Mutex<(bool, usize)>,
    /// Told when the last task under way ends.
    ended: Condvar,
    /// What a task panicked with, for the caller to hand on.
    panicked: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Helpers {
    fn new() -> Arc<Self> {
        Arc::new(Self { state: Mutex::new((false, 0)), ended: Condvar::new(), panicked: Mutex::new(None) })
    }

    /// Whether a task may begin, counted under way when it may.
    fn enter(&self) -> bool {
        let mut state = self.state.lock().expect("nothing panics holding the count");
        let (closed, under_way) = &mut *state;
        if !*closed {
            *under_way += 1;
        }
        !*closed
    }

    /// A task under way ends.
    fn leave(&self) {
        let mut state = self.state.lock().expect("nothing panics holding the count");
        state.1 -= 1;
        if state.1 == 0 {
            self.ended.notify_all();
        }
    }

    /// The step closed to the tasks yet to begin, once those under way
    /// have ended.
    fn close(&self) {
        let mut state = self.state.lock().expect("nothing panics holding the count");
        state.0 = true;
        while state.1 > 0 {
            state = self.ended.wait(state).expect("nothing panics holding the count");
        }
    }

    /// Hand on what a task panicked with, if one did.
    fn hand_on_panic(&self) {
        if let Some(panicked) = self.panicked.lock().expect("nothing panics holding a panic").take() {
            panic::resume_unwind(panicked);
        }
    }
}

/// A step the [`pool`]'s threads help with, as one of its tasks works it.
trait Step: Sync {
    fn help(&self, helpers: &Arc<Helpers>);
}

/// A task for the [`pool`] that helps with `step`, if it begins while
/// the step is open ([`Helpers`]).
fn lend<S: Step>(step: &S, helpers: &Arc<Helpers>) {
    /// The task's work, with the step's type known again.
    ///
    /// # Safety
    ///
    /// `step` is an `S` whose step is open, and stays so until `helpers`
    /// has been left.
    unsafe fn help<S: Step>(step: *const (), helpers: &Arc<Helpers>) {
        // SAFETY: the caller's.
        unsafe { &*step.cast::<S>() }.help(helpers);
    }
    let help: unsafe fn(*const (), &Arc<Helpers>) = help::<S>;
    let (step, helpers) = (Lent(std::ptr::from_ref(step).cast()), Arc::clone(helpers));
    pool().spawn(move || {
        if !helpers.enter() {
            return;
        }
        AT_WORK.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the step is open, and its caller, which holds it, waits
        // for this task to leave before it returns ([`Helpers::close`]).
        let result = panic::catch_unwind(AssertUnwindSafe(|| unsafe { help(step.get(), &helpers) }));
        AT_WORK.fetch_sub(1, Ordering::Relaxed);
        if let Err(panicked) = result {
            helpers.panicked.lock().expect("nothing panics holding a panic").get_or_insert(panicked);
        }
        helpers.leave();
    });
}

/// Where a step lies, for a task of the [`pool`] to reach it ([`lend`]).
struct Lent(*const ());

// SAFETY: it is only made of a step, which is `Sync`, and read as one.
unsafe impl Send for Lent {}

impl Lent {
    fn get(&self) -> *const () {
        self.0
    }
}

/// Closes a step to its tasks however the caller leaves it, a panic
/// included: `stop` first, so that none begins another item, and then
/// [`Helpers::close`], since those under way work on what the caller
/// holds.
struct Closing<'a, S: Fn()> {
    helpers: &'a Helpers,
    stop: S,
}

impl<S: Fn()> Drop for Closing<'_, S> {
    fn drop(&mut self) {
        (self.stop)();
        self.helpers.close();
    }
}

/// `work` done on every item by up to `threads` at once, the caller among
/// them, each answering to the request the caller answers to: the caller
/// works items as the pool's threads do, and at the end waits only for
/// those they have under way ([`Helpers`]).
fn spread<T: Sync, R: Send>(threads: usize, items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().map(work).collect();
    }
    let shared = Spread {
        items,
        work,
        next: AtomicUsize::new(0),
        done: items.iter().map(|_| Mutex::new(None)).collect(),
        unasked: AtomicUsize::new(threads - 1),
        request: current(),
    };
    let helpers = Helpers::new();
    {
        // After a panic no item more is begun.
        let _closing = Closing {
            helpers: &helpers,
            stop: || {
                shared.next.fetch_max(items.len(), Ordering::Relaxed);
            },
        };
        shared.ask(&helpers);
        shared.take();
    }
    helpers.hand_on_panic();
    shared.done.into_iter().map(|slot| slot.into_inner().expect("no worker panicked").expect("every item was worked")).collect()
}

/// What the caller of a [`spread`] and the threads that help it share.
struct Spread<'a, T, R, F> {
    items: &'a [T],
    work: F,
    /// The first item no one has begun.
    next: AtomicUsize,
    done: Vec<Mutex<Option<R>>>,
    /// Threads the step may still ask the pool for.
    unasked: AtomicUsize,
    request: Option<Cancel>,
}

impl<T: Sync, R: Send, F: Fn(&T) -> R + Sync> Spread<'_, T, R, F> {
    /// Items worked until none is left to begin.
    fn take(&self) {
        loop {
            let i = self.next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = self.items.get(i) else { break };
            let result = (self.work)(item);
            *self.done[i].lock().expect("no worker holds a result while it panics") = Some(result);
        }
    }

    /// One more thread asked for, while there are items no one has begun:
    /// a helper, once it begins, asks for the next before it works. The
    /// threads are asked for one at a time, as there proves to be work for
    /// them: every one woken goes back to sleep afterwards, at a cost in
    /// the processor's time, and a step of a few short items is done by
    /// the time a second thread would have woken.
    fn ask(&self, helpers: &Arc<Helpers>) {
        if self.next.load(Ordering::Relaxed) < self.items.len() && self.unasked.try_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1)).is_ok() {
            lend(self, helpers);
        }
    }
}

impl<T: Sync, R: Send, F: Fn(&T) -> R + Sync> Step for Spread<'_, T, R, F> {
    fn help(&self, helpers: &Arc<Helpers>) {
        self.ask(helpers);
        answering(self.request.as_ref(), || self.take());
    }
}

/// `work` done on every item, with as many of the pool's threads as
/// there are items to keep busy, the results in the items' order.
pub fn map<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    Crew::gather(items.len().saturating_sub(1)).map(items, work)
}

/// The items worked in their order until `take`, handed each result in
/// its turn, says that one settles the matter; true when one did. A loop
/// that tries one thing after another until one succeeds, with the tries
/// made side by side on the spare cores: each result is taken in its
/// turn however soon it came, so `take` sees what the loop would have
/// shown it, and once it is settled the tries still under way are called
/// off and their results thrown away.
pub fn in_turn<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync, take: impl FnMut(R) -> bool) -> bool {
    in_turn_by(usize::MAX, items, work, take)
}

/// [`in_turn`] with no more than `width` items worked at once: for work
/// that is dear in memory, where a core to spare for every item is more
/// than the machine has room for.
pub fn in_turn_by<T: Sync, R: Send>(width: usize, items: &[T], work: impl Fn(&T) -> R + Sync, take: impl FnMut(R) -> bool) -> bool {
    let crew = Crew::gather(items.len().min(width.max(1)).saturating_sub(1));
    race(crew.width(), items, work, take)
}

/// [`in_turn`], with the first item worked alone before the rest are
/// worked side by side: for a loop whose first try most often settles
/// the matter, which beside the rest would be kept waiting by them.
pub fn first_then_in_turn<T: Sync, R: Send>(items: &[T], work: impl Fn(&T) -> R + Sync, take: impl FnMut(R) -> bool) -> bool {
    first_then_in_turn_by(usize::MAX, items, work, take)
}

/// [`first_then_in_turn`] with no more than `width` of the rest worked
/// at once.
pub fn first_then_in_turn_by<T: Sync, R: Send>(
    width: usize,
    items: &[T],
    work: impl Fn(&T) -> R + Sync,
    mut take: impl FnMut(R) -> bool,
) -> bool {
    let Some((first, rest)) = items.split_first() else { return false };
    if cancelled() {
        return false;
    }
    take(work(first)) || in_turn_by(width, rest, work, take)
}

/// [`in_turn`] with `threads` items worked at once, the caller's among
/// them. The caller works an item only when it is the one it waits for
/// and the pool has not taken it up: working one past it, it would hold
/// back a result that settles the matter. It otherwise waits for the
/// item it is to take next, which a thread of the pool is then working,
/// and so never waits on a task that has not begun.
fn race<T: Sync, R: Send>(threads: usize, items: &[T], work: impl Fn(&T) -> R + Sync, mut take: impl FnMut(R) -> bool) -> bool {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().any(|item| !cancelled() && take(work(item)));
    }
    let (results, arrivals) = mpsc::channel();
    let shared = Race {
        items,
        work,
        next: AtomicUsize::new(0),
        abandoned: AtomicBool::new(false),
        // The tries answer to a request of their own, within the
        // caller's, by which those still under way are called off once
        // one settles.
        rest: Cancel::within_current(),
        results,
    };
    let helpers = Helpers::new();
    let mut settled = false;
    {
        // However the turns end, the items not yet begun are not begun,
        // and after a panic those under way are called off as well.
        let _closing = Closing {
            helpers: &helpers,
            stop: || {
                shared.abandoned.store(true, Ordering::Relaxed);
                if std::thread::panicking() {
                    shared.rest.cancel();
                }
            },
        };
        for _ in 1..threads {
            lend(&shared, &helpers);
        }
        // The caller takes the results in the items' order, holding
        // those that came early.
        let mut early: Vec<Option<R>> = items.iter().map(|_| None).collect();
        for turn in 0..items.len() {
            while early[turn].is_none() {
                if shared.next.compare_exchange(turn, turn + 1, Ordering::Relaxed, Ordering::Relaxed).is_ok() {
                    early[turn] = Some(shared.rest.scope(|| (shared.work)(&items[turn])));
                    continue;
                }
                match arrivals.recv().expect("an item taken up is sent back") {
                    (i, Ok(result)) => early[i] = Some(result),
                    (_, Err(panicked)) => panic::resume_unwind(panicked),
                }
            }
            if take(early[turn].take().expect("the result has arrived")) {
                settled = true;
                shared.rest.cancel();
                break;
            }
        }
    }
    helpers.hand_on_panic();
    settled
}

/// What the caller of a [`race`] and the threads that help it share.
struct Race<'a, T, R, F> {
    items: &'a [T],
    work: F,
    /// The first item no one has begun.
    next: AtomicUsize,
    /// Whether the turns have ended, so that no item more is begun.
    abandoned: AtomicBool,
    rest: Cancel,
    /// Each item's result, as a helper sends it back.
    results: mpsc::Sender<(usize, std::thread::Result<R>)>,
}

impl<T: Sync, R: Send, F: Fn(&T) -> R + Sync> Step for Race<'_, T, R, F> {
    fn help(&self, _: &Arc<Helpers>) {
        self.rest.scope(|| {
            // Called off by the caller's own request, every item is still
            // worked, each coming back at once with what little it has:
            // the results are waited for in order. One that panics comes
            // back as the panic, for the caller to hand on.
            while !self.abandoned.load(Ordering::Relaxed) {
                let i = self.next.fetch_add(1, Ordering::Relaxed);
                let Some(item) = self.items.get(i) else { break };
                let result = panic::catch_unwind(AssertUnwindSafe(|| (self.work)(item)));
                let failed = result.is_err();
                if self.results.send((i, result)).is_err() || failed {
                    break;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_the_items_order() {
        let items: Vec<u64> = (0..200).collect();
        // The first items take longest, so that they finish last.
        let squares = spread(8, &items, |&n| {
            std::thread::sleep(std::time::Duration::from_micros(200 - n));
            n * n
        });
        assert_eq!(squares, items.iter().map(|n| n * n).collect::<Vec<_>>());
        assert_eq!(map(&[] as &[u64], |&n| n), Vec::<u64>::new());
        assert_eq!(map(&items, |&n| n + 1)[199], 200);
        // One thread works as a loop does, and more threads than items
        // as one for each.
        assert_eq!(spread(1, &items, |&n| n + 1)[199], 200);
        assert_eq!(spread(64, &items[..3], |&n| n + 1), [1, 2, 3]);
    }

    #[test]
    fn a_crew_never_has_more_than_it_asked_for() {
        let threads = pool().current_num_threads();
        assert_eq!(Crew::gather(3).width(), 3.min(threads) + 1);
        assert_eq!(Crew::gather(usize::MAX).width(), threads + 1);
        assert_eq!(Crew::gather(0).width(), 1);
        // Other tests keep threads of the pool at work meanwhile, so of
        // the idle ones only this much is sure.
        assert!(Crew::gather_idle(3).width() <= 4);
        assert_eq!(Crew::gather_idle(0).width(), 1);
    }

    #[test]
    fn a_request_to_stop_reaches_the_work_and_the_crew() {
        assert!(!cancelled(), "work that answers to no request is never called off");
        let request = Cancel::new();
        request.scope(|| {
            assert!(!cancelled());
            request.cancel();
            assert!(cancelled());
            // The threads the work is spread over answer to it too.
            assert!(spread(4, &[1, 2, 3, 4, 5, 6, 7, 8], |_| cancelled()).into_iter().all(|called_off| called_off));
            // A scope inside answers to its own request, and the one
            // outside is in force again after it.
            Cancel::new().scope(|| assert!(!cancelled()));
            assert!(cancelled());
            // A request made for a part of the work stops for the whole's.
            assert!(Cancel::within_current().is_cancelled());
        });
        assert!(!cancelled());
        assert!(!Cancel::within_current().is_cancelled());
    }

    #[test]
    fn tries_side_by_side_are_taken_as_a_loop_would_take_them() {
        let items: Vec<u32> = (0..40).collect();
        for threads in [1, 2, 6, 64] {
            // Nothing settles it: every result is taken, in order.
            let mut seen = Vec::new();
            let slow_first = |&n: &u32| {
                std::thread::sleep(std::time::Duration::from_micros(u64::from(40 - n) * 20));
                n
            };
            assert!(!race(threads, &items, slow_first, |n| {
                seen.push(n);
                false
            }));
            assert_eq!(seen, items, "{threads} threads");
            // The eighth settles it: the seven before are taken and
            // nothing after, however many more were worked meanwhile.
            let mut seen = Vec::new();
            assert!(race(threads, &items, slow_first, |n| {
                seen.push(n);
                n == 7
            }));
            assert_eq!(seen, (0..=7).collect::<Vec<_>>(), "{threads} threads");
        }
        // Held to one at a time, the tries are a loop, and to a few, the
        // same loop's results.
        let mut seen = Vec::new();
        assert!(in_turn_by(1, &items, |&n| n, |n| {
            seen.push(n);
            n == 5
        }));
        assert_eq!(seen, (0..=5).collect::<Vec<_>>());
        assert!(first_then_in_turn_by(2, &items, |&n| n, |n| n == 20));
        assert!(!in_turn(&[] as &[u32], |&n| n, |_| true));
        assert!(!first_then_in_turn(&[] as &[u32], |&n| n, |_| true));
        // Called off, not even the first is tried.
        let request = Cancel::new();
        request.cancel();
        assert!(!request.scope(|| first_then_in_turn(&items, |&n| n, |_| true)));
        // The first alone, and nothing else worked when it settles it.
        let worked = AtomicUsize::new(0);
        let count = |&n: &u32| {
            worked.fetch_add(1, Ordering::Relaxed);
            n
        };
        assert!(first_then_in_turn(&items, count, |n| n == 0));
        assert_eq!(worked.load(Ordering::Relaxed), 1);
        assert!(first_then_in_turn(&items, count, |n| n == 39));
    }

    #[test]
    fn the_tries_still_under_way_are_called_off_once_one_settles() {
        // The first try settles it at once; the others would spin for as
        // long as nothing called them off.
        let started = std::time::Instant::now();
        let spin = |&n: &u32| {
            while n > 0 && !cancelled() && started.elapsed().as_secs() < 60 {
                std::thread::yield_now();
            }
            n
        };
        assert!(race(4, &[0, 1, 2, 3], spin, |n| n == 0));
        assert!(started.elapsed().as_secs() < 30, "the tries were not called off");
        // Called off from outside, every try comes back and is taken.
        let request = Cancel::new();
        request.cancel();
        let mut taken = 0;
        request.scope(|| race(4, &[1, 2, 3, 4], spin, |_| {
            taken += 1;
            false
        }));
        assert_eq!(taken, 4);
        assert!(!request.scope(|| race(1, &[1, 2, 3, 4], spin, |_| true)), "a loop called off tries nothing more");
    }

    /// A thread of the pool takes up the tasks of one step after those
    /// of another: each answers to the request of its own step, called
    /// off or not, and not to that of the step the thread worked before.
    #[test]
    fn a_task_answers_to_its_own_step_on_any_thread() {
        let items: Vec<u32> = (0..64).collect();
        let nested = |_: &u32| spread(4, &[0u32; 8], |_| {
            std::thread::sleep(std::time::Duration::from_micros(50));
            cancelled()
        });
        for _ in 0..20 {
            std::thread::scope(|s| {
                let called_off = s.spawn(|| {
                    let request = Cancel::new();
                    request.cancel();
                    request.scope(|| spread(8, &items, nested))
                });
                let going = s.spawn(|| spread(8, &items, nested));
                assert!(called_off.join().unwrap().iter().flatten().all(|&c| c));
                assert!(going.join().unwrap().iter().flatten().all(|&c| !c));
            });
        }
    }

    /// Steps inside steps inside steps, every kind, each waiting on its
    /// own: they finish, with every result where it belongs, and a step
    /// held to a width never has more items than that under way at once.
    #[test]
    fn steps_inside_steps_finish_and_keep_to_their_width() {
        let (under_way, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let inner = |&n: &u64| {
            let now = under_way.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            let sum: u64 = map(&[1u64, 2, 3, 4], |&k| {
                std::thread::sleep(std::time::Duration::from_micros(20));
                k * n
            })
            .into_iter()
            .sum();
            under_way.fetch_sub(1, Ordering::SeqCst);
            sum
        };
        let items: Vec<u64> = (0..12).collect();
        let outer = map(&items, |&m| {
            let mut taken = Vec::new();
            in_turn_by(3, &items, inner, |sum| {
                taken.push(sum);
                false
            });
            assert!(first_then_in_turn(&items, |&n| n * m, |p| p == 5 * m));
            taken.iter().sum::<u64>() * m
        });
        assert_eq!(outer, items.iter().map(|m| 10 * (0..12).sum::<u64>() * m).collect::<Vec<_>>());
        // Twelve outer items, each with up to three of its own inner
        // items under way.
        assert!(most.load(Ordering::SeqCst) <= 12 * 3);
        let (under_way, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        assert!(!in_turn_by(2, &items, |_| {
            let now = under_way.fetch_add(1, Ordering::SeqCst) + 1;
            most.fetch_max(now, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_micros(300));
            under_way.fetch_sub(1, Ordering::SeqCst);
        }, |()| false));
        assert!(most.load(Ordering::SeqCst) <= 2, "{} at once", most.load(Ordering::SeqCst));
    }

    /// A step on a thread of its own is not held up by another that
    /// keeps every thread of the pool at work: it works its items itself,
    /// and waits for no task of its own that has not begun, which would
    /// not begin until the other step's tries were done.
    #[test]
    fn a_step_is_not_held_up_by_another_that_keeps_the_pool_busy() {
        use std::time::{Duration, Instant};
        let items: Vec<u32> = (0..4 * (pool().current_num_threads() as u32 + 1)).collect();
        std::thread::scope(|s| {
            let other = s.spawn(|| map(&items, |_| std::thread::sleep(Duration::from_millis(300))));
            std::thread::sleep(Duration::from_millis(100));
            // No thread of the pool is idle to help.
            assert_eq!(Crew::gather_idle(usize::MAX).width(), 1);
            let started = Instant::now();
            assert_eq!(map(&[1u32, 2], |&n| n * 2), [2, 4]);
            assert!(in_turn(&[1u32, 2, 3], |&n| n, |n| n == 1));
            let mut seen = Vec::new();
            assert!(!race(3, &[1u32, 2, 3], |&n| n, |n| {
                seen.push(n);
                false
            }));
            assert_eq!(seen, [1, 2, 3]);
            // The other step's tries end a second after these began.
            assert!(started.elapsed() < Duration::from_millis(500), "{:?}", started.elapsed());
            other.join().unwrap();
        });
    }

    #[test]
    #[should_panic(expected = "an item failing, for the test")]
    fn an_item_that_panics_is_handed_on() {
        // Whichever works the item, the caller or a thread of the pool,
        // the panic comes out of the step as it was raised.
        let items: Vec<u32> = (0..64).collect();
        spread(4, &items, |&n| {
            std::thread::sleep(std::time::Duration::from_micros(100));
            if n == 40 { std::panic::resume_unwind(Box::new("an item failing, for the test")) } else { n }
        });
    }

    #[test]
    #[should_panic(expected = "a try failing, for the test")]
    fn a_try_that_panics_ends_the_wait() {
        // The result the caller waits for comes back as the panic, which
        // is handed on as it was raised.
        race(3, &[1, 2, 3, 4, 5, 6], |&n| if n == 2 { std::panic::resume_unwind(Box::new("a try failing, for the test")) } else { n }, |_| false);
    }
}
