pub struct Level {
    round_length_seconds: f64,
    required_hit_count: u32,
}

impl Level {
    pub fn new(round_length_seconds: f64, required_hit_count: u32) -> Self {
        Self {
            round_length_seconds,
            required_hit_count,
        }
    }

    pub fn round_length_seconds(&self) -> f64 {
        self.round_length_seconds
    }

    pub fn required_hit_count(&self) -> u32 {
        self.required_hit_count
    }

    pub fn make_level_list(
        initial_round_length_seconds: f64,
        round_length_decrement: f64,
        initial_required_hit_count: u32,
        required_hit_count_increment: u32,
        number_of_levels: u32,
    ) -> Vec<Self> {
        (0..number_of_levels)
            .map(|i| {
                Self::new(
                    initial_round_length_seconds - (i as f64 * round_length_decrement),
                    initial_required_hit_count + (i * required_hit_count_increment),
                )
            })
            .collect()
    }
}
