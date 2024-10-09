from utils.common import Params


class CommonParams(Params):
    def __init__(
            self,
            # simulation parameters
            sim_time_step: float = 0.05,
            sim_d_hat: float = 2e-4,
            sim_eps_d: float = 1e-4,
            sim_eps_v: float = 1e-3,
            sim_kappa: float = 1e3,
            sim_kappa_affine: float = 1e5,
            sim_kappa_con: float = 1e10,
            # simulation solver parameters
            sim_solver_newton_max_iters: int = 4,
            sim_solver_cg_max_iters: int = 50,
            sim_solver_cg_error_tolerance: float = 1e-4,
            sim_solver_cg_error_frequency=10,
            # tactile sensor parameters
            tac_sensor_meta_file: str = "gelsight_mini_e430/meta_file",
            tac_elastic_modulus_l: float = 1e6,
            tac_poisson_ratio_l: float = 0.3,
            tac_density_l: float = 1000,
            tac_elastic_modulus_r: float = 1e6,
            tac_poisson_ratio_r: float = 0.3,
            tac_density_r: float = 1000,
            tac_friction: float = 1.0,
            ccd_slackness: float = 0.7,
            ccd_thickness: float = 0.0,
            ccd_tet_inversion_thres: float = 0.0,
            ee_classify_thres: float = 1e-3,
            ee_mollifier_thres: float = 1e-3,
            allow_self_collision: bool = False,
            line_search_max_iters: int = 10,
            ccd_max_iters: int = 10,
            **kwargs
    ):
        super().__init__()
        self.sim_time_step = sim_time_step
        self.sim_d_hat = sim_d_hat
        self.sim_eps_d = sim_eps_d
        self.sim_eps_v = sim_eps_v
        self.sim_kappa = sim_kappa
        self.sim_kappa_affine = sim_kappa_affine
        self.sim_kappa_con = sim_kappa_con

        self.sim_solver_newton_max_iters = sim_solver_newton_max_iters
        self.sim_solver_cg_max_iters = sim_solver_cg_max_iters
        self.sim_solver_cg_error_tolerance = sim_solver_cg_error_tolerance
        self.sim_solver_cg_error_frequency = sim_solver_cg_error_frequency

        self.tac_sensor_meta_file = tac_sensor_meta_file
        self.tac_elastic_modulus_l = tac_elastic_modulus_l
        self.tac_poisson_ratio_l = tac_poisson_ratio_l
        self.tac_density_l = tac_density_l
        self.tac_elastic_modulus_r = tac_elastic_modulus_r
        self.tac_poisson_ratio_r = tac_poisson_ratio_r
        self.tac_density_r = tac_density_r
        self.tac_friction = tac_friction

        self.ccd_slackness = ccd_slackness
        self.ccd_thickness = ccd_thickness
        self.ccd_tet_inversion_thres = ccd_tet_inversion_thres
        self.ccd_max_iters = ccd_max_iters

        self.ee_classify_thres = ee_classify_thres
        self.ee_mollifier_thres = ee_mollifier_thres
        self.allow_self_collision = allow_self_collision

        self.line_search_max_iters = line_search_max_iters
