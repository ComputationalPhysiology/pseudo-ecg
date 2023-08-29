import cardiac_geometries


def create_slab_in_bath():
    cardiac_geometries.mesh.create_slab_in_bath(
        outdir="slab-in-bath",
        lx=1.0,
        ly=0.01,
        lz=0.5,
        bx=0.0,
        by=0.0,
        bz=0.1,
        dx=0.01,
    )


def create_slab():
    cardiac_geometries.mesh.create_slab(
        outdir="slab",
        lx=1.0,
        ly=0.01,
        lz=0.5,
        create_fibers=False,
        dx=0.01,
    )


if __name__ == "__main__":
    # create_slab_in_bath()
    create_slab()
