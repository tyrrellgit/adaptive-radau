use crate::dense_output::DenseOutput;

fn assert_close(x: f64, y: f64, tol: f64) {
    let diff = (x - y).abs();
    assert!(diff <= tol, "|{} - {}| = {} > {}", x, y, diff, tol);
}

fn make_dense(t_n: f64, h: f64) -> DenseOutput {
    DenseOutput {
        t_n,
        h,
        y_n: vec![10.0],
        c: vec![0.2, 0.7, 1.0],
        stage_vals: vec![vec![11.0], vec![15.0], vec![20.0]],
    }
}

#[test]
fn evaluate_reproduces_left_endpoint() {
    let d = make_dense(2.0, 3.0);
    assert_close(d.evaluate(2.0)[0], 10.0, 1e-12);
}

#[test]
fn evaluate_reproduces_stage_nodes() {
    let d = make_dense(2.0, 3.0);
    assert_close(d.evaluate(2.0 + 0.2 * 3.0)[0], 11.0, 1e-12);
    assert_close(d.evaluate(2.0 + 0.7 * 3.0)[0], 15.0, 1e-12);
    assert_close(d.evaluate(2.0 + 1.0 * 3.0)[0], 20.0, 1e-12);
}

#[test]
fn evaluate_at_interior_point_is_between_neighbours() {
    let d = make_dense(0.0, 1.0);
    let mid = d.evaluate(0.45)[0];
    assert!(mid > 10.0 && mid < 20.0);
}

#[test]
fn evaluate_is_continuous_across_interior() {
    let d = make_dense(0.0, 1.0);
    let left  = d.evaluate(0.499)[0];
    let right = d.evaluate(0.501)[0];
    assert!((left - right).abs() < 0.5);
}