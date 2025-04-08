import os
import onecode
import leafmap
import matplotlib.pyplot as plt
from samgeo import SamGeo, raster_to_vector


def run():
    plt.ioff()

    tiff_image = onecode.file_input(
        'Satellite Image',
        'rasters/buildings/Derna_sample.tif',
        types=[
            ('TIFF', '.tiff'),
            ('TIF', '.tif'),
            ('Geo TIFF', '.geotiff'),
            ('Geo TIF', '.geotif'),
        ]
    )

    """## Initialize SAM class

    There are several tunable parameters in automatic mask generation that control how densely points are sampled and what the thresholds are for removing low quality or duplicate masks. Additionally, generation can be automatically run on crops of the image to get improved performance on smaller objects, and post-processing can remove stray pixels and holes. Here is an example configuration that samples more masks:
    """
    onecode.Logger.info("Initialize SAM class...")
    sam_kwargs = {
        "points_per_side": onecode.number_input('Points per Side', 32, min=1),
        "pred_iou_thresh": onecode.slider('IOU threshold (prediction)', 0.86, min=0., max=1., step=0.01),
        "stability_score_thresh": onecode.slider('Stability score threshold', 0.92, min=0., max=1., step=0.01),
        "crop_n_layers": onecode.number_input('Crop N Layers', 1, min=1),
        "crop_n_points_downscale_factor": onecode.number_input('Crop N points downscale factor', 2, min=1),
        "min_mask_region_area": onecode.number_input('Minimum mask region area', 80, min=0),
    }

    sam = SamGeo(
        model_type="vit_h",
        sam_kwargs=sam_kwargs,
    )

    """## Segment the image"""
    onecode.Logger.info("Segmenting the image...")
    mask_tiff = onecode.file_output('Mask Tiff', "mask.tif")
    mask_shp = onecode.file_output('Mask Shp', "shp/mask.shp", make_path=True)
    sam.generate(tiff_image, output=mask_tiff, foreground=True)

    shp_dir = os.path.dirname(mask_shp)
    for f in os.listdir(shp_dir):
        onecode.file_output("Mask Shp", os.path.join(shp_dir, f))

    """## Convert raster to vector"""
    onecode.Logger.info("Converting raster to vector...")
    raster_to_vector(mask_tiff, output=mask_shp)

    """Display the annotations (each mask with a random color)."""
    onecode.Logger.info("Segmenting the image...")
    annotation_tiff = onecode.file_output("annotation", "annotation.tif")
    sam.show_anns(axis="off", opacity=1, output=annotation_tiff)
    plt.savefig(onecode.file_output('Annotation Image', 'annotation.png'))

    """## Compare images with a slider"""
    onecode.Logger.info("Making comparison map...")
    leafmap.image_comparison(
        tiff_image,
        annotation_tiff,
        label1="Image",
        label2="Segmentation",
        out_html=onecode.file_output(
            'Comparison',
            'comparison.html'
        )
    )

    """## Display images on an interactive map."""
    onecode.Logger.info("Exporting interactive map...")
    m = leafmap.Map(height="600px")
    m.add_basemap("SATELLITE")
    m.add_vector(mask_shp, layer_name="Vector", info_mode=None)
    m.add_layer_manager()
    m.to_html(
        onecode.file_output(
            'interactive_map',
            'interactive_map.html'
        )
    )
