"""
Milestone 9: CLI commands for coefficients
Add this to your cli.py or import it.
"""

import click
import numpy as np

from icg_cast.coefficients import registry
from icg_cast.coefficients.priors import sample_coefficient


@click.group()
def coeffs():
    """Coefficient registry commands (Milestone 9)"""
    pass


@coeffs.command("sample")
@click.option("--n", default=5, help="Number of samples to draw")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility")
def sample(n, seed):
    """Sample coefficients from their prior distributions (Milestone 9)"""
    rng = np.random.default_rng(seed)
    cards = registry()
    
    click.echo(f"Sampling {n} coefficients from priors:\n")
    
    for _i in range(n):
        name = rng.choice(list(cards.keys()))
        card = cards[name]
        median = card["default_value"]
        ev = card.get("evidence_level", "E5")
        
        val = sample_coefficient(name, median, ev, rng=rng)
        click.echo(f"  {name:45} = {val:8.4f}   (evidence={ev})")
    
    click.echo("\nDone. Use --seed for reproducibility.")


# To integrate into main CLI:
# from .cli_coeffs import coeffs
# cli.add_command(coeffs)
