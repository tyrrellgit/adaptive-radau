use crate::order_control::{OrderChange, OrderController};

#[test]
fn raises_order_after_sustained_low_iteration_counts() {
    // Smoothed iteration metric must accumulate over several steps before
    // an order raise can fire.
    let mut oc = OrderController::new(9, 5, 13);
    let mut change = OrderChange::Unchanged;
    for _ in 0..20 {
        change = oc.update(1);
        if change == OrderChange::Raised { break; }
    }
    assert_eq!(change, OrderChange::Raised);
    assert_eq!(oc.current_order, 13);
}

#[test]
fn lowers_order_after_sustained_high_iteration_counts() {
    let mut oc = OrderController::new(9, 5, 13);
    let mut change = OrderChange::Unchanged;
    for _ in 0..20 {
        change = oc.update(20);
        if change == OrderChange::Lowered { break; }
    }
    assert_eq!(change, OrderChange::Lowered);
    assert_eq!(oc.current_order, 5);
}

#[test]
fn unchanged_in_middle_band() {
    let mut oc = OrderController::new(9, 5, 13);
    assert_eq!(oc.update(5), OrderChange::Unchanged);
    assert_eq!(oc.current_order, 9);
}

#[test]
fn does_not_raise_beyond_max() {
    let mut oc = OrderController::new(13, 5, 13);
    assert_eq!(oc.update(1), OrderChange::Unchanged);
    assert_eq!(oc.current_order, 13);
}

#[test]
fn does_not_lower_below_min() {
    let mut oc = OrderController::new(5, 5, 13);
    assert_eq!(oc.update(50), OrderChange::Unchanged);
    assert_eq!(oc.current_order, 5);
}

#[test]
fn single_spike_after_low_baseline_does_not_raise_order() {
    // After converging fast (iter=1), a single slow step should not raise
    // order — it should counteract the raise signal.
    let mut oc = OrderController::new(9, 5, 13);

    // Drive histiter low
    oc.update(1);
    oc.update(1);
    oc.update(1);
    // Now a single bad step — should not raise
    let change = oc.update(9);
    assert_ne!(change, OrderChange::Raised);
}

#[test]
fn sustained_low_iterations_do_not_raise_past_max() {
    let mut oc = OrderController::new(13, 5, 13);
    for _ in 0..30 { oc.update(1); }
    assert_eq!(oc.current_order, 13);
}

#[test]
fn one_easy_step_does_not_immediately_raise_order() {
    // With histiter init = 4.0 and the EMA `0.8*hist + 0.2*iter`,
    // a single iter=1 update gives 0.8*4 + 0.2*1 = 3.4 > 2.75 → no raise.
    let mut oc = OrderController::new(9, 5, 13);
    let change = oc.update(1);
    assert_eq!(change, OrderChange::Unchanged);
    assert_eq!(oc.current_order, 9);
}