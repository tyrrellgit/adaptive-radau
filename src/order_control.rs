#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderChange {
    Raised,
    Lowered,
    Unchanged,
}

#[derive(Debug, Clone)]
pub struct OrderController {
    pub current_order: usize,
    pub histiter: f64,
    pub min_order: usize,
    pub max_order: usize,
}

impl OrderController {
    pub fn new(initial_order: usize, min_order: usize, max_order: usize) -> Self {
        Self {
            current_order: initial_order,
            histiter: 4.0,
            min_order,
            max_order,
        }
    }

    pub fn update(&mut self, iter: usize) -> OrderChange {
        self.histiter = 0.8 * iter as f64 + 0.2 * self.histiter;
        if self.histiter < 2.75 && self.current_order < self.max_order {
            self.current_order += 4;
            OrderChange::Raised
        } else if self.histiter > 8.0 && self.current_order > self.min_order {
            self.current_order -= 4;
            OrderChange::Lowered
        } else {
            OrderChange::Unchanged
        }
    }
}
