"""
Milestone 10 CLI Extension
"""

import click

from icg_cast.calibration.pipeline_milestone10 import run_milestone10_calibration


@click.command("calibrate")
@click.option("--cosmic", type=click.Path(exists=True), help="COSMIC SBS matrix")
@click.option("--toxcast", type=click.Path(exists=True), help="ToxCast summary")
@click.option("--toxcast-mapping", type=click.Path(exists=True), help="ToxCast → KCC mapping")
@click.option("--lincs", type=click.Path(exists=True), help="LINCS signatures")
@click.option("--aopwiki", type=click.Path(exists=True), help="AOP-Wiki edges")
@click.option("--apply-coefficients", is_flag=True, help="Write upgraded coefficients to disk")
@click.option("--outdir", default="outputs/calibration", help="Output directory")
def calibrate(cosmic, toxcast, toxcast_mapping, lincs, aopwiki, apply_coefficients, outdir):
    """Run Milestone 10 calibration and optionally apply coefficient updates."""
    
    result = run_milestone10_calibration(
        cosmic_path=cosmic,
        toxcast_path=toxcast,
        toxcast_mapping=toxcast_mapping,
        lincs_path=lincs,
        aopwiki_path=aopwiki,
        apply_coefficients=apply_coefficients,
        output_dir=outdir,
    )
    
    click.echo("\nMilestone 10 Calibration Complete")
    click.echo(f"  Evidence upgrades (E5 → E1-E3): {result['evidence_upgrade_count']}")
    click.echo(f"  Sources used: {', '.join(result['provenance']['sources'])}")
    
    if apply_coefficients:
        click.echo(f"\nUpgraded coefficients saved to: {outdir}/calibrated_coefficients.yaml")
        click.echo(f"Provenance saved to: {outdir}/calibration_provenance.json")
    
    return result