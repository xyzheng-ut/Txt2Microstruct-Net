import numpy as np
import bpy
import os

savefile = r""  # folder for saving stl files

numbers = 100
diameter = 0.2

for epoch in range(numbers):
    # delete models in scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    n = np.random.randint(low=25, high=35)
    seeds = np.zeros([n, 3])
    for i in range(n):
        while True:
            f = np.random.rand(1, 3)
            distance = np.sqrt(np.sum((seeds - f) ** 2, axis=-1))
            if np.all(distance > diameter):
                seeds[i] = f
                break

    for j in range(seeds.shape[0]):
        obj = bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=diameter * 0.4, location=seeds[j])

    bpy.ops.object.convert(target='MESH')
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.convert(target='MESH')

    # create a box
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(size=1., location=(0.5, 0.5, 0.5))
    bpy.ops.object.convert(target='MESH')
    cube = bpy.data.objects['Cube']
    #    obj = bpy.data.objects['obj']

    # difference boolean
    bpy.ops.object.select_pattern(pattern="Cube")
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers['Boolean'].operation = 'DIFFERENCE'
    bpy.context.object.modifiers['Boolean'].object = bpy.data.objects["Icosphere"]
    bpy.ops.object.modifier_apply(modifier="Boolean")
    for i in range(seeds.shape[0] - 1):
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.modifier_add(type='BOOLEAN')
        bpy.context.object.modifiers['Boolean'].operation = 'DIFFERENCE'
        Icosphere_name = "Icosphere." + str(i + 1).zfill(3)
        bpy.context.object.modifiers['Boolean'].object = bpy.data.objects[Icosphere_name]
        bpy.ops.object.modifier_apply(modifier="Boolean")
    #

    for i in range(seeds.shape[0] - 1):
        Icosphere_name = "Icosphere." + str(i + 1).zfill(3)
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_pattern(pattern=Icosphere_name)
        bpy.ops.object.delete()
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Icosphere")
    bpy.ops.object.delete()

    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.mesh.primitive_cube_add(size=1., location=(0.5, 0.5, 0.5))
    bpy.ops.object.convert(target='MESH')
    cube001 = bpy.data.objects['Cube.001']
    bpy.ops.object.select_pattern(pattern="Cube.001")
    bpy.ops.object.modifier_add(type='BOOLEAN')
    bpy.context.object.modifiers['Boolean'].operation = 'DIFFERENCE'
    bpy.context.object.modifiers['Boolean'].object = bpy.data.objects["Cube"]
    bpy.ops.object.modifier_apply(modifier="Boolean")

    ##    # save the mesh
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_pattern(pattern="Cube")
    bpy.ops.object.delete()
    target_file = os.path.join(savefile, '%d.stl' % epoch)
    ##    print(target_file)
    bpy.context.scene.objects['Cube.001'].select_set(True)
    bpy.ops.export_mesh.stl(filepath=target_file)