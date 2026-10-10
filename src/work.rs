//! Calling a slow step off, and spreading one over the cores that stand
//! idle.
//!
//! A search of a photo climbs a ladder of copies and fits a placement to
//! every outline the finder patterns propose; a read tries every kernel
//! under both threshold rules, and then the photo's other three turns. On
//! a sound symbol the first try settles it. On a damaged one, which is
//! what this crate is for, every try is made and fails, one after another
//! on one core, for minutes. Two things mend that.
//!
//! The tries a step would make in turn are made side by side, on the
//! cores nothing else is using, and taken in the order they would have
//! been made in, so that what a step finds is what it found on one core.
//! The first try is still made alone wherever it most often settles the
//! matter: made beside the rest, it would have to wait for them.
//!
//! And whoever started a step may call it off: the loops that take the
//! time ask [`cancelled`] as they go round, and come back with what they
//! have. The request to stop is not handed down through every function
//! between the caller and those loops. It is set for the thread the work
//! runs on, by [`Cancel::scope`], and handed on to every thread the work
//! is spread over.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, mpsc};

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
        let _restore = Restore(CURRENT.with(|current| current.replace(Some(self.clone()))));
        work()
    }
}

fn current() -> Option<Cancel> {
    CURRENT.with(|current| current.borrow().clone())
}

/// Whether the work this thread is doing has been called off. Work that
/// answers to no request is never called off.
pub fn cancelled() -> bool {
    CURRENT.with(|current| current.borrow().as_ref().is_some_and(Cancel::is_cancelled))
}

/// The cores no crew has taken: every core but one at the start, the one
/// being the caller's own.
fn spare() -> &'static AtomicUsize {
    static SPARE: OnceLock<AtomicUsize> = OnceLock::new();
    SPARE.get_or_init(|| AtomicUsize::new(std::thread::available_parallelism().map_or(1, |n| n.get()).saturating_sub(1)))
}

/// Spare cores taken for one step, and given back when it is dropped. A
/// step that finds none spare, as one of many running side by side does,
/// has a crew of itself alone and works as it would on one core: the
/// cores are shared out over the whole process, so that steps inside
/// steps do not start more threads than there are cores to run them.
pub struct Crew {
    extra: usize,
}

impl Crew {
    /// Take up to `most` of the spare cores.
    pub fn gather(most: usize) -> Self {
        let spare = spare();
        let mut free = spare.load(Ordering::Relaxed);
        loop {
            let take = free.min(most);
            if take == 0 {
                return Self { extra: 0 };
            }
            match spare.compare_exchange_weak(free, free - take, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => return Self { extra: take },
                Err(now) => free = now,
            }
        }
    }

    /// How many items the crew works at once: the caller and the cores
    /// taken.
    pub fn width(&self) -> usize {
        self.extra + 1
    }

    /// `work` done on every item, the results in the items' order
    /// whichever was finished first.
    pub fn map<T: Sync, R: Send>(&self, items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
        spread(self.width(), items, work)
    }
}

impl Drop for Crew {
    fn drop(&mut self) {
        spare().fetch_add(self.extra, Ordering::AcqRel);
    }
}

/// `work` done on every item by up to `threads` threads, the caller's
/// among them, each answering to the request the caller answers to.
fn spread<T: Sync, R: Send>(threads: usize, items: &[T], work: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().map(work).collect();
    }
    let next = AtomicUsize::new(0);
    let done: Vec<Mutex<Option<R>>> = items.iter().map(|_| Mutex::new(None)).collect();
    let take = || {
        loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            let Some(item) = items.get(i) else { break };
            let result = work(item);
            *done[i].lock().expect("no worker holds a result while it panics") = Some(result);
        }
    };
    let request = current();
    std::thread::scope(|scope| {
        for _ in 1..threads {
            scope.spawn(|| match &request {
                Some(request) => request.scope(take),
                None => take(),
            });
        }
        take();
    });
    done.into_iter().map(|slot| slot.into_inner().expect("no worker panicked").expect("every item was worked")).collect()
}

/// `work` done on every item, on as many spare cores as there are items
/// to keep busy, the results in the items' order.
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

/// [`in_turn`] on `threads` threads, the caller's among them.
fn race<T: Sync, R: Send>(threads: usize, items: &[T], work: impl Fn(&T) -> R + Sync, mut take: impl FnMut(R) -> bool) -> bool {
    let threads = threads.min(items.len());
    if threads <= 1 {
        return items.iter().any(|item| !cancelled() && take(work(item)));
    }
    // The tries answer to a request of their own, within the caller's,
    // by which those still under way are called off once one settles.
    let rest = Cancel::within_current();
    let abandoned = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    let (results, arrivals) = mpsc::channel::<(usize, R)>();
    let mut settled = false;
    std::thread::scope(|scope| {
        for _ in 0..threads {
            let results = results.clone();
            let (rest, abandoned, next, work) = (&rest, &abandoned, &next, &work);
            scope.spawn(move || {
                rest.scope(|| {
                    // Called off by the caller's own request, every item
                    // is still worked, each coming back at once with what
                    // little it has: the results are waited for in order.
                    while !abandoned.load(Ordering::Relaxed) {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(item) = items.get(i) else { break };
                        if results.send((i, work(item))).is_err() {
                            break;
                        }
                    }
                });
            });
        }
        drop(results);
        // The caller takes the results in the items' order, holding
        // those that came early. A worker that panicked sends nothing,
        // and the channel closes when the last worker ends: the wait
        // ends there, and the scope hands the panic on.
        let mut early: Vec<Option<R>> = items.iter().map(|_| None).collect();
        'turns: for turn in 0..items.len() {
            while early[turn].is_none() {
                match arrivals.recv() {
                    Ok((i, result)) => early[i] = Some(result),
                    Err(_) => break 'turns,
                }
            }
            if take(early[turn].take().expect("the result has arrived")) {
                settled = true;
                break;
            }
        }
        abandoned.store(true, Ordering::Relaxed);
        if settled {
            rest.cancel();
        }
    });
    settled
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
        // Other tests gather crews of their own meanwhile, so only this
        // much is sure.
        assert!(Crew::gather(3).width() <= 4);
        assert_eq!(Crew::gather(0).width(), 1);
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

    #[test]
    #[should_panic(expected = "a scoped thread panicked")]
    fn a_try_that_panics_ends_the_wait() {
        // The result the caller waits for never comes: the wait ends when
        // the other tries have, and the panic is handed on.
        race(3, &[1, 2, 3, 4, 5, 6], |&n| if n == 2 { std::panic::resume_unwind(Box::new("a try failing, for the test")) } else { n }, |_| false);
    }
}
