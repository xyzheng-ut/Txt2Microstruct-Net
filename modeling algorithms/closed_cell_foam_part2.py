import numpy as np
import os, re
import bpy


def load_region_facet_point(file=r"face\face_0.txt"):
    points = []
    region = []
    facet = []
    with open(file, "r") as f:
        x = f.readlines()
        for tline in x:
            if tline[0] == "p":
                temp = tline.split(',')
                x = float(temp[1])
                y = float(temp[2])
                z = float(temp[3])
                points.append([x, y, z])
            elif tline[0] == "f":
                temp = re.findall(r"\d+", tline[:])[1:]
                facet.append([int(x) for x in temp])
            elif tline[0] == "r":
                temp = re.findall(r"\d+", tline[:])[1:]
                region.append([int(x) for x in temp])
    return region, facet, points


def return_facet_point(region, facets, points):
    new_point_list = []
    new_facet_list = []
    for facet_index in region:
        new_facet = []
        for point_index in facets[facet_index]:
            if point_index not in new_point_list:
                new_point_list.append(point_index)
            new_facet.append(new_point_list.index(point_index))
        new_facet_list.append(new_facet)
    new_point_ultimate = []
    for i in new_point_list:
        new_point_ultimate.append(points[i])

    return np.array(new_facet_list), np.array(new_point_ultimate)


def find_center(vertices):
    vertices = np.reshape(vertices, [-1, 3])
    max_x = np.ndarray.max(vertices.T[0])
    max_y = np.ndarray.max(vertices.T[1])
    max_z = np.ndarray.max(vertices.T[2])
    min_x = np.ndarray.min(vertices.T[0])
    min_y = np.ndarray.min(vertices.T[1])
    min_z = np.ndarray.min(vertices.T[2])
    center = [(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2]

    return center


def position_list_to_model(loadfile, savefile):
    files = os.listdir(loadfile)
    for file in files:
        scale_set = np.random.uniform(low=0.8, high=0.95)

        # delete models in scene
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

        # load file
        file_dir = loadfile + '\\' + file
        num = [int(index) for index in file.split('.') if index.isdigit()][0]
        regions, facets, points = load_region_facet_point(file_dir)

        # build each polyhedron
        for epoch in range(len(regions)):

            # define vertices and loops
            region = regions[epoch]
            edges, vertices = return_facet_point(region, facets, points)
            center = find_center(vertices)
            vertices = np.array(vertices)
            edges = np.array(edges, dtype=object)
            num_vertices = vertices.shape[0]
            vertex_index = edges
            loop_total = []
            for i in vertex_index:
                loop_total.append(len(i))
            loop_total = np.array(loop_total)
            loop_start = []
            temp = 0
            for i in loop_total:
                loop_start.append(temp)
                temp += i
            loop_start = np.array(loop_start)

            vertex_index_new = []
            for i in vertex_index:
                for j in i:
                    vertex_index_new.append(j)
            num_vertex_indices = len(vertex_index_new)
            num_loops = len(loop_start)

            # Create mesh object based on the arrays above
            mesh = bpy.data.meshes.new(name='created mesh_%d' % epoch)
            mesh.vertices.add(num_vertices)
            mesh.vertices.foreach_set("co", np.ndarray.flatten(vertices))
            mesh.loops.add(num_vertex_indices)
            mesh.loops.foreach_set("vertex_index", np.array(vertex_index_new))
            mesh.polygons.add(num_loops)
            mesh.polygons.foreach_set("loop_start", np.ndarray.flatten(loop_start))
            mesh.polygons.foreach_set("loop_total", np.ndarray.flatten(loop_total))

            # We're done setting up the mesh values, update mesh object and
            # let Blender do some checks on it
            mesh.update()
            mesh.validate()

            # Create Object whose Object Data is our new mesh
            obj = bpy.data.objects.new('created object_%d' % epoch, mesh)

            # Add *Object* to the scene, not the mesh
            scene = bpy.context.scene
            scene.collection.objects.link(obj)

            # Select the new object and make it active
            bpy.ops.object.select_all(action='DESELECT')
            bpy.ops.object.select_pattern(pattern="created object_%d" % epoch)

            # shrink the polyhedron
            scale = scale_set  # 0.88 = rho0.5, 0.95=rho0.15
            bpy.ops.transform.resize(value=(scale, scale, scale), center_override=center, remove_on_cancel=True)

        # Select all mesh objects
        ALL_OBJS = [m for m in bpy.context.scene.objects if m.type == 'MESH']
        for OBJS in ALL_OBJS:
            OBJS.select_set(state=True)
            bpy.context.view_layer.objects.active = OBJS

        # join all objects and set box array
        bpy.ops.object.select_all(action='SELECT')
        obj.select_set(True)
        bpy.ops.object.join()
        bpy.ops.object.duplicate()
        bpy.ops.transform.translate(value=(1, 0, 0))

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        bpy.ops.object.duplicate()
        bpy.ops.transform.translate(value=(0, 1, 0))

        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()
        bpy.ops.object.duplicate()
        bpy.ops.transform.translate(value=(0, 0, 1))
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.join()

        # convert objects to mesh
        bpy.ops.object.select_all(action='SELECT')
        obj = bpy.context.selected_objects

        for obj in bpy.data.objects:
            obj.name = 'obj'

        bpy.ops.object.select_pattern(pattern="obj")
        bpy.ops.object.convert(target='MESH')

        # weld the mesh
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.mesh.tris_convert_to_quads()
        bpy.ops.mesh.normals_make_consistent()
        bpy.ops.object.editmode_toggle()

        # create a box
        bpy.ops.mesh.primitive_cube_add(size=1., location=(3, 3, 3))
        bpy.ops.object.convert(target='MESH')
        cube = bpy.data.objects['Cube']
        obj = bpy.data.objects['obj']

        # difference boolean
        bpy.ops.object.select_pattern(pattern="Cube")
        bpy.ops.object.modifier_add(type='BOOLEAN')
        bpy.context.object.modifiers['Boolean'].operation = 'DIFFERENCE'
        bpy.context.object.modifiers['Boolean'].object = obj
        bpy.ops.object.modifier_apply(modifier="Boolean")

        # save the mesh
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['obj'].select_set(True)
        bpy.ops.object.select_pattern(pattern="obj")
        bpy.ops.object.delete()
        target_file = os.path.join(savefile, '%d.stl' % num)
        print(target_file)
        bpy.context.scene.objects['Cube'].select_set(True)
        bpy.ops.export_mesh.stl(filepath=target_file)


# main
loadfile = r""  # folder of seeds
savefile = r""  # folder for saving stl files
position_list_to_model(loadfile, savefile)
