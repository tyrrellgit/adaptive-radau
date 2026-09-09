use nalgebra::DMatrix;

use crate::error::RadauError;
use crate::tableau::{RadauTableau, TableauCache};

fn assert_close(x: f64, y: f64, tol: f64) {
    let diff = (x - y).abs();
    assert!(diff <= tol, "|{} - {}| = {} > {}", x, y, diff, tol);
}

#[test]
fn supported_orders_have_expected_stage_counts() {
    let t5  = RadauTableau::for_order(5).unwrap();
    let t9  = RadauTableau::for_order(9).unwrap();
    let t13 = RadauTableau::for_order(13).unwrap();

    assert_eq!(t5.order,  5);  assert_eq!(t5.stages,  3);
    assert_eq!(t9.order,  9);  assert_eq!(t9.stages,  5);
    assert_eq!(t13.order, 13); assert_eq!(t13.stages, 7);
}

#[test]
fn invalid_order_errors() {
    assert!(matches!(RadauTableau::for_order(4),  Err(RadauError::InvalidOrder(4))));
    assert!(matches!(RadauTableau::for_order(6),  Err(RadauError::InvalidOrder(6))));
}

#[test]
fn last_collocation_node_is_one() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        assert_close(*t.c.last().unwrap(), 1.0, 1e-14);
    }
}

#[test]
fn b_equals_last_row_of_a() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        for i in 0..t.stages {
            assert_close(t.a[(t.stages - 1, i)], t.b[i], 1e-14);
        }
    }
}

/// Stage order condition: A·c^(k−1) = c^k / k for k = 1..s.
/// This is the defining property of a collocation method.
#[test]
fn collocation_conditions_hold() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let s = t.stages;
        for k in 1..=s {
            for i in 0..s {
                let lhs: f64 = (0..s).map(|j| t.a[(i, j)] * t.c[j].powi((k - 1) as i32)).sum();
                let rhs = t.c[i].powi(k as i32) / k as f64;
                assert_close(lhs, rhs, 1e-12);
            }
        }
    }
}

/// Quadrature order conditions for b: sum b_i c_i^(k−1) = 1/k for k = 1..=2s−1.
/// This is the classical (2s−1)-order property of Radau IIA.
#[test]
fn b_quadrature_is_order_2s_minus_1() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let s = t.stages;
        for k in 1..=2 * s - 1 {
            let lhs: f64 = (0..s).map(|i| t.b[i] * t.c[i].powi((k - 1) as i32)).sum();
            assert_close(lhs, 1.0 / k as f64, 1e-12);
        }
    }
}

/// A and A_inv satisfy A * A_inv == I.
#[test]
fn a_times_a_inv_is_identity() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let prod = &t.a * &t.a_inv;
        let id = DMatrix::<f64>::identity(t.stages, t.stages);
        for i in 0..t.stages {
            for j in 0..t.stages {
                assert_close(prod[(i, j)], id[(i, j)], 1e-10);
            }
        }
    }
}

/// u1 (real eigenvalue of A^{-1}) matches the classical reference value
/// for the 3-stage tableau:
///   u1 = 30 / (6 + 81^(1/3) − 9^(1/3))
#[test]
fn u1_matches_closed_form_for_3_stage() {
    let t = RadauTableau::for_order(5).unwrap();
    let u1_ref = 30.0 / (6.0 + 81.0_f64.cbrt() - 9.0_f64.cbrt());
    assert_close(t.u1, u1_ref, 1e-12);
}

/// Hairer's ESTRAD DD coefficients must satisfy the moment system
///   DD_s        = -1/s
///   sum DD_j c_j = -1
///   sum DD_j c_j^k = 0 for k = 2..s-1
/// These guarantee leading-order cancellation against f(y_n) so that the
/// embedded error is O(h^s).
#[test]
fn dd_satisfies_moment_conditions() {
    for order in [5, 9, 13] {
        let t = RadauTableau::for_order(order).unwrap();
        let s = t.stages;

        // Normalisation.
        assert_close(t.dd[s - 1], -1.0 / s as f64, 1e-12);

        // First moment.
        let m1: f64 = (0..s).map(|j| t.dd[j] * t.c[j]).sum();
        assert_close(m1, -1.0, 1e-10);

        // Zero moments for k = 2..s-1.
        for k in 2..s {
            let mk: f64 = (0..s).map(|j| t.dd[j] * t.c[j].powi(k as i32)).sum();
            assert_close(mk, 0.0, 1e-9);
        }
    }
}

/// For the 3-stage tableau, Hairer's published DD values are
///   DD_1 = -(13 + 7√6)/3,  DD_2 = (-13 + 7√6)/3,  DD_3 = -1/3.
#[test]
fn dd_matches_hairer_published_values_3_stage() {
    let t = RadauTableau::for_order(5).unwrap();
    let sqrt6 = 6.0_f64.sqrt();
    let dd1_ref = -(13.0 + 7.0 * sqrt6) / 3.0;
    let dd2_ref = (-13.0 + 7.0 * sqrt6) / 3.0;
    let dd3_ref = -1.0 / 3.0;
    assert_close(t.dd[0], dd1_ref, 1e-10);
    assert_close(t.dd[1], dd2_ref, 1e-10);
    assert_close(t.dd[2], dd3_ref, 1e-12);
}

#[test]
fn cache_returns_same_pointer_for_same_order() {
    let mut cache = TableauCache::default();
    let p1 = cache.get(5).unwrap() as *const _;
    let p2 = cache.get(5).unwrap() as *const _;
    assert_eq!(p1, p2);
}

#[test]
fn cache_holds_multiple_orders_independently() {
    let mut cache = TableauCache::default();
    let p5  = cache.get(5).unwrap()  as *const _;
    let p9  = cache.get(9).unwrap()  as *const _;
    let p13 = cache.get(13).unwrap() as *const _;
    assert_ne!(p5, p9);
    assert_ne!(p9, p13);
}
