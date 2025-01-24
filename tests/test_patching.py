



_patch = """diff --git a/scene/__init__.py b/scene/__init__.py
index 2b31398..c31679a 100644
--- a/scene/__init__.py
+++ b/scene/__init__.py
@@ -25 +25 @@ class Scene:
-    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
+    def __init__(self, args : ModelParams, scene_info, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
@@ -43,8 +42,0 @@ class Scene:
-        if os.path.exists(os.path.join(args.source_path, "sparse")):
-            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
-        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
-            print("Found transforms_train.json file, assuming Blender data set!")
-            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
-        else:
-            assert False, "Could not recognize scene type!"
-
@@ -52,2 +43,0 @@ class Scene:
-            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
-                dest_file.write(src_file.read())
@@ -62,2 +51,0 @@ class Scene:
-            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
-                json.dump(json_cams, file)
"""

def test_parse_patch():
    from nerfbaselines.methods._patching import _parse_patch

    parsed = _parse_patch(_patch)
    print(parsed)
    assert parsed is not None


def test_apply_patch():
    before = "1\n2\n3\n4\n5\n6\n7\n8\n9\n"
    after = "passed\n1\n2\ntest\ntest2\n3\n4\n5\n8\n9\n10\n11\n"
    patch = """diff --git a/a b/a
index c0ce4d7..804fbf9 100644
--- a/a
+++ b/a
@@ -1,3 +1,6 @@
+passed
 1
 2
+test
+test2
 3
@@ -5,6 +8,6 @@
 5
-6
-7
 8
 9
+10
+11
 
"""
    from nerfbaselines.methods._patching import _parse_patch, _apply_patch
    p = _parse_patch(patch)
    assert "a" in p
    updates = p["a"]
    assert _apply_patch(before, updates) == after

    before = "1\n2\n3\n4\n5\n6\n7\n8\n9"
    after = "passed\n1\n2\ntest\ntest2\n3\n4\n5\n8\n9\n10\n11"
    patch = """diff --git a/a b/a
index c0ce4d7..804fbf9 100644
--- a/a
+++ b/a
@@ -1,3 +1,6 @@
+passed
 1
 2
+test
+test2
 3
@@ -5,5 +8,5 @@
 5
-6
-7
 8
 9
+10
+11
"""
    p = _parse_patch(patch)
    assert "a" in p
    updates = p["a"]
    assert _apply_patch(before, updates) == after
