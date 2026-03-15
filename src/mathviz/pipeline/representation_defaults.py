"""Default representation configs per generator name."""

from mathviz.core.representation import RepresentationConfig, RepresentationType

_TUBE_CONFIG = RepresentationConfig(type=RepresentationType.TUBE, tube_radius=0.05)
_SURFACE_CONFIG = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
_SPARSE_CONFIG = RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
_HEIGHTMAP_CONFIG = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
_KNOT_TUBE_CONFIG = RepresentationConfig(
    type=RepresentationType.TUBE, tube_radius=0.1
)

GENERATOR_DEFAULTS: dict[str, RepresentationConfig] = {
    # Parametric surfaces (mesh)
    "torus": _SURFACE_CONFIG,
    "klein_bottle": _SURFACE_CONFIG,
    "sphere": _SURFACE_CONFIG,
    "boy_surface": _SURFACE_CONFIG,
    "costa_surface": _SURFACE_CONFIG,
    "enneper_surface": _SURFACE_CONFIG,
    "lissajous_surface": _SURFACE_CONFIG,
    "mobius_strip": _SURFACE_CONFIG,
    "mobius_trefoil": _SURFACE_CONFIG,
    "linked_tori": _SURFACE_CONFIG,
    "twisted_torus": _SURFACE_CONFIG,
    "rose_surface": _SURFACE_CONFIG,
    "spherical_harmonics": _SURFACE_CONFIG,
    "superellipsoid": _SURFACE_CONFIG,
    "generic_parametric": _SURFACE_CONFIG,
    "calabi_yau": _SURFACE_CONFIG,
    "cross_cap": _SURFACE_CONFIG,
    "dini_surface": _SURFACE_CONFIG,
    "dupin_cyclide": _SURFACE_CONFIG,
    "roman_surface": _SURFACE_CONFIG,
    "seifert_surface": _SURFACE_CONFIG,
    "bour_surface": _SURFACE_CONFIG,
    "dna_helix": _TUBE_CONFIG,
    "hopf_fibration": _TUBE_CONFIG,
    # Geometry (curves)
    "geodesic_sphere": _SURFACE_CONFIG,
    "voronoi_sphere": _TUBE_CONFIG,
    # Implicit surfaces (mesh)
    "genus2_surface": _SURFACE_CONFIG,
    "gyroid": _SURFACE_CONFIG,
    "schwarz_d": _SURFACE_CONFIG,
    "schwarz_p": _SURFACE_CONFIG,
    # Data-driven mesh
    "building_extrude": _SURFACE_CONFIG,
    "parabolic_envelope": _SURFACE_CONFIG,
    # Attractors (curves)
    "lorenz": _TUBE_CONFIG,
    "rossler": _TUBE_CONFIG,
    "aizawa": _TUBE_CONFIG,
    "chen": _TUBE_CONFIG,
    "clifford": _TUBE_CONFIG,
    "dequan_li": _TUBE_CONFIG,
    "double_pendulum": _TUBE_CONFIG,
    "halvorsen": _TUBE_CONFIG,
    "sprott": _TUBE_CONFIG,
    "thomas": _TUBE_CONFIG,
    # Knots (curves)
    "torus_knot": _KNOT_TUBE_CONFIG,
    "figure_eight_knot": _KNOT_TUBE_CONFIG,
    "lissajous_knot": _KNOT_TUBE_CONFIG,
    "seven_crossing_knots": _KNOT_TUBE_CONFIG,
    "pretzel_knot": _KNOT_TUBE_CONFIG,
    "cinquefoil_knot": _KNOT_TUBE_CONFIG,
    "borromean_rings": _KNOT_TUBE_CONFIG,
    "chain_links": _KNOT_TUBE_CONFIG,
    "trefoil_on_torus": _KNOT_TUBE_CONFIG,
    # Curves
    "lissajous": _TUBE_CONFIG,
    "lissajous_curve": _TUBE_CONFIG,
    "cardioid": _TUBE_CONFIG,
    "fibonacci_spiral": _TUBE_CONFIG,
    "hilbert_3d": _TUBE_CONFIG,
    "logarithmic_spiral": _TUBE_CONFIG,
    "voronoi_3d": _TUBE_CONFIG,
    "soundwave": _TUBE_CONFIG,
    # Physics (curves + surfaces)
    "kepler_orbit": _TUBE_CONFIG,
    "nbody": _TUBE_CONFIG,
    "planetary_positions": _TUBE_CONFIG,
    "electron_orbital": _SURFACE_CONFIG,
    "magnetic_field": _TUBE_CONFIG,
    "gravitational_lensing": _TUBE_CONFIG,
    "wave_interference": _SURFACE_CONFIG,
    # Number theory (point clouds)
    "sacks_spiral": _SPARSE_CONFIG,
    "prime_gaps": _SPARSE_CONFIG,
    "ulam_spiral": _SPARSE_CONFIG,
    "digit_encoding": _SPARSE_CONFIG,
    # Fractals (sparse shell from mesh)
    "mandelbulb": _SPARSE_CONFIG,
    "julia3d": _SPARSE_CONFIG,
    "menger_sponge": _SURFACE_CONFIG,
    "sierpinski_tetrahedron": _SURFACE_CONFIG,
    "apollonian_3d": _SURFACE_CONFIG,
    "quaternion_julia": _SURFACE_CONFIG,
    "ifs_fractal": _SPARSE_CONFIG,
    "koch_3d": _SURFACE_CONFIG,
    # Heightmaps (scalar field)
    "mandelbrot": _HEIGHTMAP_CONFIG,
    "mandelbrot_heightmap": _HEIGHTMAP_CONFIG,
    "burning_ship": _HEIGHTMAP_CONFIG,
    "fractal_slice": _HEIGHTMAP_CONFIG,
    "heightmap": _HEIGHTMAP_CONFIG,
    "noise_surface": _HEIGHTMAP_CONFIG,
    "reaction_diffusion": _HEIGHTMAP_CONFIG,
    "terrain": _HEIGHTMAP_CONFIG,
    # Procedural
    "lsystem": _TUBE_CONFIG,
    "rd_surface": _SURFACE_CONFIG,
    "penrose_3d": _SURFACE_CONFIG,
    "weaire_phelan": _TUBE_CONFIG,
}
