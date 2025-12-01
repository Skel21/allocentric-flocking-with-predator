"""
Run multiple simulation scenarios comparing allocentric vs egocentric navigation
"""
import numpy as np
from simulation import create_agent_group, run_simulation


def run_single_scenario(
    prey_allocentric: bool,
    predator_allocentric: bool,
    scenario_name: str
):
    """Run one simulation scenario with specified parameters.
    
    Args:
        prey_allocentric: Whether prey use allocentric coordinates
        predator_allocentric: Whether predators use allocentric coordinates
        scenario_name: Name for this scenario
    """
    print(f"\n{'='*60}")
    print(f"Running scenario: {scenario_name}")
    print(f"  Prey: {'Allocentric' if prey_allocentric else 'Egocentric'}")
    print(f"  Predator: {'Allocentric' if predator_allocentric else 'Egocentric'}")
    print(f"{'='*60}\n")
    
    # Simulation parameters
    NUM_TICKS = 200
    
    # Prey configuration
    NUM_PREY = 10
    PREY_STEPS = 100
    PREY_VISIBILITY = 500
    PREY_BETA = 10
    PREY_UNFRIENDLY_WEIGHT = -(NUM_PREY + 2)
    
    # Target configuration
    NUM_TARGETS = 1
    TARGET_WEIGHT = NUM_PREY + 1
    targets = [np.random.uniform(-20, 40, size=2) for _ in range(NUM_TARGETS)]
    
    # Predator configuration
    NUM_PREDATORS = 1
    PREDATOR_STEPS = 100
    PREDATOR_VISIBILITY = 500
    PREDATOR_BETA = 10
    PREDATOR_WEIGHT = 3
    
    # Create agent groups
    prey = create_agent_group(
        num_agents=NUM_PREY,
        steps_per_tick=PREY_STEPS,
        is_allocentric=prey_allocentric,
        visibility_radius=PREY_VISIBILITY,
        beta=PREY_BETA,
        target_weight=TARGET_WEIGHT,
        unfriendly_weight=PREY_UNFRIENDLY_WEIGHT
    )
    
    predators = create_agent_group(
        num_agents=NUM_PREDATORS,
        steps_per_tick=PREDATOR_STEPS,
        is_allocentric=predator_allocentric,
        visibility_radius=PREDATOR_VISIBILITY,
        beta=PREDATOR_BETA,
        target_weight=PREDATOR_WEIGHT
    )
    
    # Run simulation
    output_dir = f"simulations/{scenario_name}"
    run_simulation(
        num_ticks=NUM_TICKS,
        targets=targets,
        prey_group=prey,
        predator_group=predators,
        output_dir=output_dir,
        show_progress=True
    )
    
    print(f"Target positions: {targets}")


def main():
    """Run all four experimental conditions."""
    scenarios = [
        (True, True, "both_allocentric"),
        (True, False, "prey_allo_pred_ego"),
        (False, True, "prey_ego_pred_allo"),
        (False, False, "both_egocentric")
    ]
    
    for prey_allo, pred_allo, name in scenarios:
        run_single_scenario(prey_allo, pred_allo, name)
    
    print("\n" + "="*60)
    print("All simulations complete!")
    print("="*60)


if __name__ == "__main__":
    main()