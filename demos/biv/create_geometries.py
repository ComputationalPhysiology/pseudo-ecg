import cardiac_geometries


def create_biv_in_torso():

    cardiac_geometries.mesh.create_biv_ellipsoid_torso(
        outdir="biv-in-torso",
        char_length=0.2,
        heart_as_surface=False,
        torso_length=8.0,
        torso_width=8.0,
        torso_height=8.0,
        rotation_angle=0.5235987755982988,
        center_lv_y=0.0,
        center_lv_z=0.0,
        a_endo_lv=2.5,
        b_endo_lv=1.0,
        c_endo_lv=1.0,
        a_epi_lv=3.0,
        b_epi_lv=1.5,
        c_epi_lv=1.5,
        center_rv_y=0.5,
        center_rv_z=0.0,
        a_endo_rv=3.0,
        b_endo_rv=1.5,
        c_endo_rv=1.5,
        a_epi_rv=4.0,
        b_epi_rv=2.5,
        c_epi_rv=2.0,
        create_fibers=True,
        fiber_angle_endo=-60,
        fiber_angle_epi=60,
        fiber_space="P_1",
    )


def create_biv():
    cardiac_geometries.mesh.create_biv_ellipsoid(
        outdir="biv",
        char_length=0.1,
        center_lv_y=0.0,
        center_lv_z=0.0,
        a_endo_lv=2.5,
        b_endo_lv=1.0,
        c_endo_lv=1.0,
        a_epi_lv=3.0,
        b_epi_lv=1.5,
        c_epi_lv=1.5,
        center_rv_y=0.5,
        center_rv_z=0.0,
        a_endo_rv=3.0,
        b_endo_rv=1.5,
        c_endo_rv=1.5,
        a_epi_rv=4.0,
        b_epi_rv=2.5,
        c_epi_rv=2.0,
        create_fibers=True,
        fiber_angle_endo=-60,
        fiber_angle_epi=60,
        fiber_space="P_1",
    )


if __name__ == "__main__":
    create_biv_in_torso()
    create_biv()
