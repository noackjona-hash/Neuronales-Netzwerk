use neuro_racer::evolution::{EvolutionConfig, Population};
use neuro_racer::track::Track;

#[test]
fn test_evolution_convergence() {
    let track = Track::preset_super_speedway();
    let config = EvolutionConfig {
        population_size: 50,
        elitism_count: 5,
        mutation_rate: 0.10,
        mutation_strength: 0.25,
        tournament_size: 4,
        max_generation_time: 25.0,
        ..Default::default()
    };

    let mut pop = Population::new(config, &track, 42);
    let dt = 1.0 / 60.0;

    let initial_best = pop.current_best_fitness();

    // Run 5 generations
    for _gen in 0..5 {
        while !pop.is_generation_over() {
            pop.step(dt, &track);
        }
        pop.advance_generation(&track);
    }

    assert!(pop.generation == 6);
    assert!(pop.best_ever_fitness >= initial_best);
    println!("Evolution test passed. Best fitness achieved: {:.2}", pop.best_ever_fitness);
}
