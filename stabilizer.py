import cv2
import math
import numpy as np
import statistics
import tqdm
import sys
class Stabilizer:

    '''
    Enum indicating which definition to use for the energy function's adaptive weights.

    The values are:
    * ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL: Calculate the adaptive weights using the linear
        model presented in the original paper.
    * ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED: Calculate the adaptive weights using a variant of the
        original model in which one of the terms has had its sign flipped.

    * ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH: Set the adaptive weights to a constant high
        value.
    * ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW: Set the adaptive weights to a constant low value.
        This model is based on the authors' claim that smaller adaptive weights lead to less
        cropping and wobbling. Here both terms in the energy equation have equal weight.
    '''

    ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL = 0
    ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED = 1
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH = 2
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW = 3


    # Los valores constantes de los pesos adaptativos altos y bajos.
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE = 100
    ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE = 1


    def __init__(self, mesh_row_count=16, mesh_col_count=16,
        mesh_outlier_subframe_row_count=4, mesh_outlier_subframe_col_count=4,
        feature_ellipse_row_count=10, feature_ellipse_col_count=10,
        homography_min_number_corresponding_features=4,
        temporal_smoothing_radius=10, optimization_num_iterations=100,
        color_outside_image_area_bgr=(0, 0, 255),
        visualize=False):
        '''
        Constructor.

        Input:

        * mesh_row_count: The number of rows contained in the mesh.
            NOTE There are 1 + mesh_row_count vertices per row.
        * mesh_col_count: The number of cols contained in the mesh.
            NOTE There are 1 + mesh_col_count vertices per column.
        * mesh_outlier_subframe_row_count: The height in rows of each subframe when breaking down
            the image into subframes to eliminate outlying features.
        * mesh_outlier_subframe_col_count: The width of columns of each subframe when breaking
            down the image into subframes to eliminate outlying features.
        * feature_ellipse_row_count: The height in rows of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * feature_ellipse_col_count: The width in columns of the ellipse drawn around each feature
            to match it with vertices in the mesh.
        * homography_min_number_corresponding_features: The minimum number of features that must
            correspond between two frames to perform a homography.
        * temporal_smoothing_radius: In the energy function used to smooth the image, the number of
            frames to inspect both before and after each frame when computing that frame's
            regularization term. Thus, the regularization term involves a sum over up to
            2 * temporal_smoothing_radius frame indexes.
            NOTE This constant is denoted as \Omega_{t} in the original paper.
        * optimization_num_iterations: The number of iterations of the Jacobi method to perform when
            minimizing the energy function.
        * color_outside_image_area_bgr: The color, expressed in BGR, to display behind the
            stabilized footage in the output.
            NOTE This color should be removed during cropping, but is customizable just in case.
        * visualize: Whether or not to display a video loop of the unstabilized and cropped,
            stabilized videos after saving the stabilized video. Pressing Q closes the window.

        Output:

        (A Stabilizer object.)
        '''

        self.mesh_col_count = mesh_col_count
        self.mesh_row_count = mesh_row_count
        self.mesh_outlier_subframe_row_count = mesh_outlier_subframe_row_count
        self.mesh_outlier_subframe_col_count = mesh_outlier_subframe_col_count
        self.feature_ellipse_row_count = feature_ellipse_row_count
        self.feature_ellipse_col_count = feature_ellipse_col_count
        self.homography_min_number_corresponding_features = homography_min_number_corresponding_features
        self.temporal_smoothing_radius = temporal_smoothing_radius
        self.optimization_num_iterations = optimization_num_iterations
        self.color_outside_image_area_bgr = color_outside_image_area_bgr
        self.visualize = visualize

        self.feature_detector = cv2.FastFeatureDetector_create()


    def stabilize(self, input_path, output_path, adaptive_weights_definition=ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL):
        '''
        Read in the video at the given input path and output a stabilized version to the given
        output path.

        Input:

        * input_path: The path to a video.
        * output_path: The path where the stabilized version of the video should be placed.
        * adaptive_weights_definition: Which method to use for computing the energy function's adaptive
            weights.

        Output:

        (The stabilized video is saved to output_path.)

        In addition, the function returns a tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            stability scores.
        '''

        if not (adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or
                adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED or
                adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH or
                adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW):
            raise ValueError(
                'Invalid value for `adaptive_weights_definition`. Expecting value of '
                '`Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL`, '
                '`Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED`, '
                '`Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH`, or'
                '`Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW`.'
            )

        unstabilized_frames, num_frames, frames_per_second, codec = self._get_unstabilized_frames_and_video_features(input_path)
        vertex_unstabilized_displacements_by_frame_index, homographies = self._get_unstabilized_vertex_displacements_and_homographies(num_frames, unstabilized_frames)
        vertex_stabilized_displacements_by_frame_index = self._get_stabilized_vertex_displacements(
            num_frames, unstabilized_frames, adaptive_weights_definition,
            vertex_unstabilized_displacements_by_frame_index, homographies
        )
        stabilized_frames, crop_boundaries = self._get_stabilized_frames_and_crop_boundaries(
            num_frames, unstabilized_frames,
            vertex_unstabilized_displacements_by_frame_index,
            vertex_stabilized_displacements_by_frame_index
        )
        cropped_frames = self._crop_frames(stabilized_frames, crop_boundaries)

        cropping_ratio, distortion_score = self._compute_cropping_ratio_and_distortion_score(num_frames, unstabilized_frames, cropped_frames)
        stability_score = self._compute_stability_score(num_frames, vertex_stabilized_displacements_by_frame_index)

        self._write_stabilized_video(output_path, num_frames, frames_per_second, codec, cropped_frames)

        if self.visualize:
            self._display_unstablilized_and_cropped_video_loop(num_frames, frames_per_second, unstabilized_frames, cropped_frames)

        return (cropping_ratio, distortion_score, stability_score)


    def _get_unstabilized_frames_and_video_features(self, input_path):
        '''
        Helper method for stabilize.
        Return each frame of the input video as a NumPy array along with miscellaneous video
        features.

        Input:

        * input_path: The path to the unstabilized video.

        Output:

        A tuple of the following items in order.

        * unstabilized_frames: A list of the frames in the unstabilized video, each represented as a
            NumPy array.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        '''

        unstabilized_video = cv2.VideoCapture(input_path)
        num_frames = int(unstabilized_video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_per_second = unstabilized_video.get(cv2.CAP_PROP_FPS)
        codec = int(unstabilized_video.get(cv2.CAP_PROP_FOURCC))

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Reading video from <{input_path}>')

            unstabilized_frames = []
            for frame_index in t:
                unstabilized_frame = self._get_next_frame(unstabilized_video)
                if unstabilized_frame is None:
                    raise IOError(
                        f'Video at <{input_path}> did not have frame {frame_index} of '
                        f'{num_frames} (indexed from 0).'
                    )
                unstabilized_frames.append(unstabilized_frame)

        unstabilized_video.release()

        return (unstabilized_frames, num_frames, frames_per_second, codec)


    def _get_next_frame(self, video):
        '''
        Helper method for _get_unstabilized_frames_and_video_features.

        Return the next frame of the given video.

        Input:

        * video: A VideoCapture object.

        Output:

        * next_frame: If available, the next frame in the video as a NumPy array, and None
            otherwise.
        '''

        frame_successful, pixels = video.read()
        return pixels if frame_successful else None


    def _get_unstabilized_vertex_displacements_and_homographies(self, num_frames, unstabilized_frames):
        '''
        Helper method for stabilize.
        Return the displacements for the given unstabilized frames.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * vertex_unstabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the unstabilized displacements of each vertex in the  mesh.
            In particular,
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the x-displacement of the mesh vertex at the given row and col from frame 0 to
            frame frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        * homographies: A NumPy array of shape
            (num_frames, 3, 3)
            containing global homographies between frames.
            In particular, homographies[frame_index] contains a homography matrix between frames
            frame_index and frame_index + 1 (that is, the homography to construct frame_index + 1).
            Since no frame comes after num_frames - 1,
            homographies[num_frames-1] is the identity homography.
        '''

        vertex_unstabilized_displacements_by_frame_index = np.empty(
            (num_frames, self.mesh_row_count + 1, self.mesh_col_count + 1, 2)
        )
        vertex_unstabilized_displacements_by_frame_index[0].fill(0)

        homographies = np.empty((num_frames, 3, 3))
        homographies[-1] = np.identity(3)

        with tqdm.trange(num_frames - 1) as t:
            t.set_description('Computing unstabilized mesh displacements')
            for current_index in t:
                current_frame, next_frame = unstabilized_frames[current_index:current_index+2]
                current_velocity, homography = self._get_unstabilized_vertex_velocities(current_frame, next_frame)
                vertex_unstabilized_displacements_by_frame_index[current_index+1] = vertex_unstabilized_displacements_by_frame_index[current_index] + current_velocity
                homographies[current_index] = homography

        return (vertex_unstabilized_displacements_by_frame_index, homographies)


    def _get_unstabilized_vertex_velocities(self, early_frame, late_frame):
        '''
        Helper method for _get_unstabilized_vertex_displacements_and_homographies.

        Given two adjacent frames (the "early" and "late" frames), estimate the velocities of the
        vertices in the early frame.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.

        Output:

        A tuple of the following items in order.

        * mesh_velocities: A NumPy array of shape
            (mesh_row_count + 1, mesh_col_count + 1, 2)
            where the entry mesh_velocities[row][col][0]
            contains the x-velocity of the mesh vertex at the given row and col during early_frame,
            and mesh_velocities[row][col][1] contains the corresponding y-velocity.
            NOTE since time is discrete and in units of frames, a vertex's velocity during
            early_frame is the same as its displacement from early_frame to late_frame.
        * early_to_late_homography: A NumPy array of shape (3, 3) representing the homography
            between early_frame and late_frame.
        '''

        # aplicar esta homografía a una coordenada en el marco temprano mapea a donde estará
        # En el marco tardío, suponiendo que el punto no se someta a movimiento
        early_features, late_features, early_to_late_homography = self._get_matched_features_and_homography(early_frame, late_frame)

        # Cada vértice comenzó en el cuadro temprano en una posición dada por Vértex_X_Y_BY_ROW_COLAND.
        # Si no tiene velocidad en relación con la escena (es decir, el vértice está temblando con él), entonces a
        # Obtenga su posición en el marco tardío, aplicamos a los primeros
        # posición.
        # El desplazamiento entre estas posiciones es su movimiento global.
        frame_height, frame_width, = early_frame.shape[:2]
        vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)
        #AplicarUnaTransformaciónDePerspectivaALasCoordenadasDeLosVértices
        #UtilizandoLaMatrizDeHomografíaEarlyToLateHomography
        #Luego,CalcularLasVelocidadesGlobalesDeLosVérticesRestandoLasCoordenadas
        #OriginalesDeLasCoordenadasTransformadas
        vertex_global_velocities = cv2.perspectiveTransform(vertex_x_y, early_to_late_homography) - vertex_x_y
        vertex_global_velocities_by_row_col = np.reshape(vertex_global_velocities, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
        vertex_global_x_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 0]
        vertex_global_y_velocities_by_row_col = vertex_global_velocities_by_row_col[:, :, 1]

        # Además del movimiento anterior (que mueve cada vértice a su lugar en la malla en
        # Late_frame), cada vértice puede sufrir un movimiento residual adicional para que coincida con su cercano
        # características.
        # Después de reunir estas velocidades, realice el primer filtro mediano:
        # Ordene las velocidades de cada vértice por componente X, luego por componente Y, y use la mediana
        # elemento como la velocidad del vértice.
        vertex_nearby_feature_residual_x_velocities_by_row_col, vertex_nearby_feature_residual_y_velocities_by_row_col = self._get_vertex_nearby_feature_residual_velocities(frame_width, frame_height, early_features, late_features, early_to_late_homography)

        vertex_residual_x_velocities_by_row_col = np.array([
            [
                statistics.median(x_velocities)
                if x_velocities else 0
                for x_velocities in row
            ]
            for row in vertex_nearby_feature_residual_x_velocities_by_row_col
        ])
        vertex_residual_y_velocities_by_row_col = np.array([
            [
                statistics.median(y_velocities)
                if y_velocities else 0
                for y_velocities in row
            ]
            for row in vertex_nearby_feature_residual_y_velocities_by_row_col
        ])
        vertex_x_velocities_by_row_col = (vertex_global_x_velocities_by_row_col + vertex_residual_x_velocities_by_row_col).astype(np.float32)
        vertex_y_velocities_by_row_col = (vertex_global_y_velocities_by_row_col + vertex_residual_y_velocities_by_row_col).astype(np.float32)

        # Realizar el segundo filtro mediano:
        # Reemplace la velocidad de cada vértice con la velocidad media de sus vecinos.
        vertex_smoothed_x_velocities_by_row_col = cv2.medianBlur(vertex_x_velocities_by_row_col, 3)
        vertex_smoothed_y_velocities_by_row_col = cv2.medianBlur(vertex_y_velocities_by_row_col, 3)
        vertex_smoothed_velocities_by_row_col = np.dstack((vertex_smoothed_x_velocities_by_row_col, vertex_smoothed_y_velocities_by_row_col))
        return (vertex_smoothed_velocities_by_row_col, early_to_late_homography)


    def _get_vertex_nearby_feature_residual_velocities(self, frame_width, frame_height, early_features, late_features, early_to_late_homography):
        '''
        Helper method for _get_unstabilized_vertex_velocities.

        Given two adjacent frames, return a list that maps each vertex in the mesh to the residual
        velocities of its nearby features.

        Input:

        * frame_width: the width of the windows' frames.
        * frame_height: the height of the windows' frames.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            frame, not the window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion

        Output:

        A tuple of the following items in order.

        * vertex_nearby_feature_x_velocities_by_row_col: A list
            where entry vertex_nearby_feature_x_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        * vertex_nearby_feature_y_velocities_by_row_col: A list
            where entry vertex_nearby_feature_y_velocities_by_row_col[row, col] contains a list of
            the x-velocities of all the features nearby the vertex at the given row and col.
        '''

        vertex_nearby_feature_x_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]
        vertex_nearby_feature_y_velocities_by_row_col = [
            [[] for _ in range(self.mesh_col_count + 1)]
            for _ in range(self.mesh_row_count + 1)
        ]

        if early_features is not None:
            # Calcule las velocidades de las características;Ver https://stackoverflow.com/a/44409124 para
            # Combinar las posiciones y velocidades en una matriz

            # Si un punto no hubiera sufrido movimiento, entonces su posición en el marco tardío sería
            # Se encuentra aplicando Early_TO_Late_Homography a su posición en el cuadro temprano.
            # El movimiento adicional del punto es lo que lo lleva de esa posición a su
            # posición.
            feature_residual_velocities = late_features - cv2.perspectiveTransform(early_features, early_to_late_homography)
            feature_positions_and_residual_velocities = np.c_[early_features, feature_residual_velocities]

            # Aplicar las velocidades de las características a los vértices de malla cercana
            for feature_position_and_residual_velocity in feature_positions_and_residual_velocities:
                feature_x, feature_y, feature_residual_x_velocity, feature_residual_y_velocity = feature_position_and_residual_velocity[0]
                feature_row = (feature_y / frame_height) * self.mesh_row_count
                feature_col = (feature_x / frame_width) * self.mesh_col_count

                # Dibuja una elipse alrededor de cada característica
                # de ancho self.feature_ellipse_col_count
                # y altura self.feature_ellipse_row_count,
                # y aplique la velocidad de la función a todos los vértices de malla que caen dentro de este
                # Ellipse.
                # Para hacer esto, podemos iterar a través de todas las filas que cubre la Elipse.
                # Para cada fila, podemos usar la ecuación para una elipse centrada en el
                # característica para determinar qué columnas cubre las elipse.El resultante
                #(fila, columna) Los pares corresponden a los vértices en la elipse.
                ellipse_top_row_inclusive = max(0, math.ceil(feature_row - self.feature_ellipse_row_count / 2))
                ellipse_bottom_row_exclusive = 1 + min(self.mesh_row_count, math.floor(feature_row + self.feature_ellipse_row_count / 2))

                for vertex_row in range(ellipse_top_row_inclusive, ellipse_bottom_row_exclusive):

                    # medio ancho derivado de la ecuación de elipse
                    ellipse_slice_half_width = self.feature_ellipse_col_count * math.sqrt((1/4) - ((vertex_row - feature_row) / self.feature_ellipse_row_count) ** 2)
                    ellipse_left_col_inclusive = max(0, math.ceil(feature_col - ellipse_slice_half_width))
                    ellipse_right_col_exclusive = 1 + min(self.mesh_col_count, math.floor(feature_col + ellipse_slice_half_width))

                    for vertex_col in range(ellipse_left_col_inclusive, ellipse_right_col_exclusive):
                        vertex_nearby_feature_x_velocities_by_row_col[vertex_row][vertex_col].append(feature_residual_x_velocity)
                        vertex_nearby_feature_y_velocities_by_row_col[vertex_row][vertex_col].append(feature_residual_y_velocity)

        return (vertex_nearby_feature_x_velocities_by_row_col, vertex_nearby_feature_y_velocities_by_row_col)


    def _get_matched_features_and_homography(self, early_frame, late_frame):
        '''
        Helper method for _get_unstabilized_vertex_velocities and _compute_cropping_ratio_and_distortion_score.

        Detect features in the early window using the Stabilizer's feature_detector
        and track them into the late window using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_frame: A NumPy array representing the frame before late_frame.
        * late_frame: A NumPy array representing the frame after early_frame.

        Output:

        A tuple of the following items in order.

        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_frame that was
            successfully tracked in late_frame. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in late_frame that was
            successfully tracked from early_frame. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        * early_to_late_homography: A homography matrix that maps a point in early_frame to its
            corresponding location in late_frame, assuming the point is not undergoing motion.
            If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_to_late_homography is None.
        '''

        # Obtenga características que se han eliminado los valores atípicos aplicando homografías a los subframas

        frame_height, frame_width = early_frame.shape[:2]
        subframe_width = math.ceil(frame_width / self.mesh_outlier_subframe_col_count)
        subframe_height = math.ceil(frame_height / self.mesh_outlier_subframe_row_count)

        # Early_Feature_By_Subframe [i] contiene una matriz CV_32FC2 de las características tempranas en el
        # marco i^th subtrame;
        # late_features_by_subframe [i] se define de manera similar
        early_features_by_subframe = []
        late_features_by_subframe = []

        # Todos paralelizan
        for subframe_left_x in range(0, frame_width, subframe_width):
            for subframe_top_y in range(0, frame_height, subframe_height):
                early_subframe = early_frame[subframe_top_y:subframe_top_y+subframe_height,
                                           subframe_left_x:subframe_left_x+subframe_width]
                late_subframe = late_frame[subframe_top_y:subframe_top_y+subframe_height,
                                         subframe_left_x:subframe_left_x+subframe_width]
                subframe_offset = [subframe_left_x, subframe_top_y]
                subframe_early_features, subframe_late_features = self._get_features_in_subframe(
                    early_subframe, late_subframe, subframe_offset
                )
                if subframe_early_features is not None:
                    early_features_by_subframe.append(subframe_early_features)
                if subframe_late_features is not None:
                    late_features_by_subframe.append(subframe_late_features)

        early_features = np.concatenate(early_features_by_subframe)
        late_features = np.concatenate(late_features_by_subframe)

        if len(early_features) < self.homography_min_number_corresponding_features:
            return (None, None, None)

        early_to_late_homography, _ = cv2.findHomography(
            early_features, late_features
        )

        return (early_features, late_features, early_to_late_homography)


    def _get_features_in_subframe(self, early_subframe, late_subframe, subframe_offset):
        '''
        Helper method for _get_matched_features_and_homography.
        Track and return features that appear between the two given frames, eliminating outliers
        by applying a homography using RANSAC.

        Input:

        * early_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_subframe.
        * late_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_subframe.
        * offset_location: A tuple (x, y) representing the offset of the subframe within its frame,
            relative to the frame's top left corner.

        Output:

        A tuple of the following items in order.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            frame, not the subframe. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each non-outlying feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            frame, not the subframe. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        '''

        # Reúna todas las características que rastrean entre marcos
        early_features_including_outliers, late_features_including_outliers = self._get_all_matched_features_between_subframes(early_subframe, late_subframe)
        if early_features_including_outliers is None:
            return (None, None)

        # Eliminar características periféricas con Ransac
        _, outlier_features = cv2.findHomography(
            early_features_including_outliers, late_features_including_outliers, method=cv2.RANSAC
        )
        outlier_features_mask = outlier_features.flatten().astype(dtype=bool)
        early_features = early_features_including_outliers[outlier_features_mask]
        late_features = late_features_including_outliers[outlier_features_mask]

        # Agregue un desplazamiento constante para presentar coordenadas para expresarlas
        # en relación con la esquina superior izquierda del marco original, no la subtrama
        return (early_features + subframe_offset, late_features + subframe_offset)


    def _get_all_matched_features_between_subframes(self, early_subframe, late_subframe):
        '''
        Helper method for _get_feature_positions_in_subframe.
        Detect features in the early subframe using the Stabilizer's feature_detector
        and track them into the late subframe using cv2.calcOpticalFlowPyrLK.

        Input:

        * early_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame before late_subframe.
        * late_subframe: A NumPy array (or a view into one) representing a subsection of the pixels
            in the frame after early_subframe.

        Output:

        A tuple of the following items in order.
        * early_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in early_subframe that was
            successfully tracked in late_subframe. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            early_features is None.
        * late_features: A CV_32FC2 array (see https://stackoverflow.com/a/47617999) of positions
            containing the coordinates of each feature in late_subframe that was
            successfully tracked from early_subframe. These coordinates are expressed relative to the
            window. If fewer than
            self.homography_min_number_corresponding_features such features were found,
            late_features is None.
        '''

        # Convierta una lista de punto clave en una matriz CV_32FC2 que contenga las coordenadas de cada punto clave;
        # Consulte https://stackoverflow.com/a/55398871 y https://stackoverflow.com/a/47617999
        early_keypoints = self.feature_detector.detect(early_subframe)
        if len(early_keypoints) < self.homography_min_number_corresponding_features:
            return (None, None)

        early_features_including_unmatched = np.float32(cv2.KeyPoint_convert(early_keypoints)[:, np.newaxis, :])
        late_features_including_unmatched, matched_features, _ = cv2.calcOpticalFlowPyrLK(
            early_subframe, late_subframe, early_features_including_unmatched, None
        )

        matched_features_mask = matched_features.flatten().astype(dtype=bool)
        early_features = early_features_including_unmatched[matched_features_mask]
        late_features = late_features_including_unmatched[matched_features_mask]

        if len(early_features) < self.homography_min_number_corresponding_features:
            return (None, None)

        return (early_features, late_features)


    def _get_stabilized_vertex_displacements(self, num_frames, unstabilized_frames, adaptive_weights_definition, vertex_unstabilized_displacements_by_frame_index, homographies):
        '''
        Helper method for stabilize.

        Return each vertex's displacement at each frame in the stabilized video.

        Specifically, find the displacements that minimize an energy function.
        The energy function takes displacements as input and outputs a number corresponding
        to how how shaky the input is.

        The output array of stabilized displacements is calculated using the
        Jacobi method. For each mesh vertex, the method solves the equation
        A p = b
        for vector p,
        where entry p[i] contains the vertex's stabilized displacement at frame i.
        The entries in matrix A and vector b were derived by finding the partial derivative of the
        energy function with respect to each p[i] and setting them all to 0. Thus, solving for p in
        A p = b results in displacements that produce a local extremum (which we can safely
        assume is a local minimum) in the energy function.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized displacements of each vertex in the  mesh, as outputted
            by _get_unstabilized_frames_and_video_features.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * vertex_stabilized_displacements_by_frame_index: A NumPy array of shape
            (num_frames, self.mesh_row_count, self.mesh_col_count, 2)
            containing the stabilized displacements of each vertex in the  mesh.
            In particular,
            vertex_stabilized_displacements_by_frame_index[frame_index][row][col][0]
            contains the x-displacement (the x-displacement in addition to any imposed by
            global homographies) of the mesh vertex at the given row and col from frame 0 to frame
            frame_index, both inclusive.
            vertex_unstabilized_displacements_by_frame_index[frame_index][row][col][1]
            contains the corresponding y-displacement.
        '''


        frame_height, frame_width = unstabilized_frames[0].shape[:2]

        off_diagonal_coefficients, on_diagonal_coefficients = self._get_jacobi_method_input(num_frames, frame_width, frame_height, adaptive_weights_definition, homographies)

        # vértex_unstabilized_dispplacements_by_frame_index está indexado por
        # frame_index, luego fila, luego col, luego componente de velocidad.
        # En su lugar, vertex_unstabilized_dispplacements_by_coord está indexado por
        # fila, luego col, luego frame_index, luego componente de velocidad;
        # Este reordenamiento debe permitir un acceso más rápido durante el paso de optimización.
        vertex_unstabilized_displacements_by_coord = np.moveaxis(
            vertex_unstabilized_displacements_by_frame_index, 0, 2
        )
        vertex_stabilized_displacements_by_coord = np.empty(vertex_unstabilized_displacements_by_coord.shape)
        # Todos paralelizan
        with tqdm.trange((self.mesh_row_count + 1) * (self.mesh_col_count + 1)) as t:
            t.set_description('Computing stabilized mesh displacements')
            for mesh_coords_flattened in t:
                mesh_row = mesh_coords_flattened // (self.mesh_row_count + 1)
                mesh_col = mesh_coords_flattened % (self.mesh_col_count + 1)
                vertex_unstabilized_displacements = vertex_unstabilized_displacements_by_coord[mesh_row][mesh_col]
                vertex_stabilized_displacements = self._get_jacobi_method_output(
                    off_diagonal_coefficients, on_diagonal_coefficients,
                    vertex_unstabilized_displacements,
                    vertex_unstabilized_displacements
                )
                vertex_stabilized_displacements_by_coord[mesh_row][mesh_col] = vertex_stabilized_displacements

            vertex_stabilized_displacements_by_frame_index = np.moveaxis(
                vertex_stabilized_displacements_by_coord, 2, 0
            )

        return vertex_stabilized_displacements_by_frame_index


    def _get_jacobi_method_input(self, num_frames, frame_width, frame_height, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_stabilized_displacements.
        The Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximates a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.
        Return the values in matrix A given the video's features and the user's chosen method.

        Input:

        * num_frames: The number of frames in the video.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        A tuple of the following items in order.

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
        * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
        '''

        # row_indexes [fila] [col] = row, col_indexes [fila] [col] = col
        row_indexes, col_indexes = np.indices((num_frames, num_frames))

        # regularización_palos [t, r] es una constante de peso aplicada al término de regularización.
        # En el artículo, regularización_palos [t, r] se denota como w_ {t, r}.
        # Tenga en cuenta que regularization_weights [i, i] = 0.
        regularization_weights = np.exp(
            -np.square((3 / self.temporal_smoothing_radius) * (row_indexes - col_indexes))
        )

        # adaptive_weights [t] es un peso, derivado de las propiedades de los marcos, aplicados a
        # Término de regularización correspondiente al marco en el índice t
        # Tenga en cuenta que el documento no especifica el peso para aplicarse al último cuadro (que no
        # Tener una velocidad), por lo que asumimos que es lo mismo que el segundo a los marco.
        # En el documento, adaptive_weights [t] se denota como \ lambda_ {t}.
        adaptive_weights = self._get_adaptive_weights(num_frames, frame_width, frame_height, adaptive_weights_definition, homographies)
        # adaptive_weights = np.full ((num_frames,), 10)

        # combined_adaptive_regularization_weaks [t, r] = \ lambda_ {t} w_ {t, r}
        combined_adaptive_regularization_weights = np.matmul(np.diag(adaptive_weights), regularization_weights)

        # La entrada fuera de diagonal en Cell [T, R] está escrita como
        #-2 * \ lambda_ {t} w_ {t, r}
        off_diagonal_coefficients = -2 * combined_adaptive_regularization_weights

        # La entrada en diagonal en la celda [t, t] está escrita como
        # 1 + 2 * \ sum_ {r \ in \ omega_ {t}, r \ neq t} \ lambda_ {t} w_ {t, r}.
        # Nota ya que w_ {t, t} = 0,
        # Podemos ignorar la restricción r \ neq t en la suma y escribir la entrada en diagonal en
        # celda [t, t] como
        # 1 + 2 * \ sum {r \ in \ omega_ {t}} \ lambda_ {t} w_ {t, r}.
        on_diagonal_coefficients = 1 + 2 * np.sum(combined_adaptive_regularization_weights, axis=1)

        # Establezca coeficientes en 0 para T, r, r;Ver https://stackoverflow.com/a/36247680
        off_diagonal_mask = np.zeros(off_diagonal_coefficients.shape)
        for i in range(-self.temporal_smoothing_radius, self.temporal_smoothing_radius + 1):
            off_diagonal_mask += np.diag(np.ones(num_frames - abs(i)), i)
        off_diagonal_coefficients = np.where(off_diagonal_mask, off_diagonal_coefficients, 0)

        return (off_diagonal_coefficients, on_diagonal_coefficients)


    def _get_adaptive_weights(self, num_frames, frame_width, frame_height, adaptive_weights_definition, homographies):
        '''
        Helper method for _get_jacobi_method_input.
        Return the array of adaptive weights for use in the energy function.

        Input:

        * num_frames: The number of frames in the video.
        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.
        * adaptive_weights_definition: Which definition to use for the energy function's adaptive
            weights.
        * homographies: A NumPy array of homographies as generated by
            _get_unstabilized_vertex_displacements_and_homographies.

        Output:

        * adaptive_weights: A NumPy array of size
            (num_frames,).
            adaptive_weights[t] is a weight, derived from properties of the frames, applied to the
            regularization term corresponding to the frame at index t.
            Note that the paper does not specify the weight to apply to the last frame (which does
            not have a velocity), so we assume it is the same as the second-to-last frame.
            In the paper, adaptive_weights[t] is denoted as \lambda_{t}.
        '''

        if adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL or adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_FLIPPED:
            # Los pesos adaptativos se determinan conectando los valores propios de cada homografía
            # componente afín en un modelo lineal
            homography_affine_components = homographies.copy()
            homography_affine_components[:, 2, :] = [0, 0, 1]
            adaptive_weights = np.empty((num_frames,))

            for frame_index in range(num_frames):
                homography = homography_affine_components[frame_index]
                sorted_eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(homography)))

                translational_element = math.sqrt((homography[0, 2] / frame_width) ** 2 + (homography[1, 2] / frame_height) ** 2)
                affine_component = sorted_eigenvalue_magnitudes[-2] / sorted_eigenvalue_magnitudes[-1]

                adaptive_weight_candidate_1 = -1.93 * translational_element + 0.95

                if adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL:
                    adaptive_weight_candidate_2 = 5.83 * affine_component + 4.88
                else:  # Adaptive_weaws_definition_flipped
                    adaptive_weight_candidate_2 = 5.83 * affine_component - 4.88

                adaptive_weights[frame_index] = max(
                    min(adaptive_weight_candidate_1, adaptive_weight_candidate_2), 0
                )
        elif adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH:
            adaptive_weights = np.full((num_frames,), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_HIGH_VALUE)
        elif adaptive_weights_definition == Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW:
            adaptive_weights = np.full((num_frames,), self.ADAPTIVE_WEIGHTS_DEFINITION_CONSTANT_LOW_VALUE)

        return adaptive_weights


    def _get_jacobi_method_output(self, off_diagonal_coefficients, on_diagonal_coefficients, x_start, b):
        '''
        Helper method for _get_stabilized_displacements.
        Using the Jacobi method (see https://en.wikipedia.org/w/index.php?oldid=1036645158),
        approximate a solution for the vector x in the equation
        A x = b
        where A is a matrix of constants and b is a vector of constants.

        Return a value of x after performing self.optimization_num_iterations of the Jacobi method.

        Input:

        * off_diagonal_coefficients: A 2D NumPy array containing the off-diagonal entries of A.
            Specifically, off_diagonal_coefficients[i, j] = A_{i, j} where i != j, and all
            on-diagonal entries off_diagonal_coefficients[i, i] = 0.
            In the Wikipedia link, this matrix is L + U.
         * on_diagonal_coefficients: A 1D NumPy array containing the on-diagonal entries of A.
            Specifically, on_diagonal_coefficients[i] = A_{i, i}.
            In the Wikipedia link, this array is the diagonal entries of D.
        * x_start: A NumPy array containing an initial estimate for x.
        * b: A NumPy array containing the constant vector b.

        Output:

        * x: A NumPy array containing the value of x computed with the Jacobi method.
        '''

        x = x_start.copy()

        reciprocal_on_diagonal_coefficients_matrix = np.diag(np.reciprocal(on_diagonal_coefficients))

        for _ in range(self.optimization_num_iterations):
            x = np.matmul(reciprocal_on_diagonal_coefficients_matrix, b - np.matmul(off_diagonal_coefficients, x))

        return x


    def _get_vertex_x_y(self, frame_width, frame_height):
        '''
        Helper method for _get_stabilized_frames_and_crop_boundaries_and_crop_boundaries and _get_unstabilized_vertex_velocities.
        Return a NumPy array that maps [row, col] coordinates to [x, y] coordinates.

        Input:

        * frame_width: the width of the video's frames.
        * frame_height: the height of the video's frames.

        Output:

        row_col_to_vertex_x_y: A CV_32FC2 array (see https://stackoverflow.com/a/47617999)
            containing the coordinates [x, y] of vertices in the mesh. This array is ordered
            so that when this array is reshaped to
            (self.mesh_row_count + 1, self.mesh_col_count + 1, 2),
            the resulting entry in [row, col] contains the coordinates [x, y] of the vertex in the
            top left corner of the cell at the mesh's given row and col.
        '''

        return np.array([
            [[math.ceil((frame_width - 1) * (col / (self.mesh_col_count))),
              math.ceil((frame_height - 1) * (row / (self.mesh_row_count)))]]
            for row in range(self.mesh_row_count + 1)
            for col in range(self.mesh_col_count + 1)
        ], dtype=np.float32)


    def _get_stabilized_frames_and_crop_boundaries(self, num_frames, unstabilized_frames, vertex_unstabilized_displacements_by_frame_index, vertex_stabilized_displacements_by_frame_index):
        '''
        Helper method for stabilize.

        Return stabilized copies of the given unstabilized frames warping them according to the
        given transformation data, as well as boundaries representing how to crop these stabilized
        frames.

        Input:

        * num_frames: The number of frames in the unstabilized video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * vertex_unstabilized_displacements_by_frame_index: A NumPy array containing the
            unstabilized displacements of each vertex in the  mesh, as generated by
            _get_unstabilized_vertex_displacements_and_homographies.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized displacements of each vertex in the  mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        A tuple of the following items in order.

        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.
        '''

        frame_height, frame_width = unstabilized_frames[0].shape[:2]

        # unstabilized_vertex_x_y y stabilized_vertex_x_y son CV_32FC2 Numpy Arrays
        #(Ver https://stackoverflow.com/a/47617999)
        # de las coordenadas de los nodos de malla en el video estabilizado, indexado desde la parte superior izquierda
        # esquina y moverse de izquierda a derecha, de arriba a abajo.
        unstabilized_vertex_x_y = self._get_vertex_x_y(frame_width, frame_height)

        # row_col_to_unstabilized_vertex_x_y [fila, col] y
        # row_col_to_stabilized_vertex_x_y [fila, col]
        # contener las posiciones x e y del vértice en la fila dada y col.
        row_col_to_unstabilized_vertex_x_y = np.reshape(unstabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))

        # stabilized_motion_mesh_by_frame_index [frame_index] es una matriz Numpy CV_32FC2
        #(ver https://stackoverflow.com/a/47617999) que contiene la cantidad a agregar a cada vértice
        # Coordinar para transformarlo de su posición no estabilizada en Frame_Index a su
        # Posición estabilizada en Frame_index.
        # Dado que los desplazamientos actuales están dados por
        # VERTEX_UNSTABILIZED_DISPLACEMENTS [Frame_index],
        # y los desplazamientos finales están dados por
        # vertex_stabilized_displacements [frame_index], agregando la diferencia de las dos
        # produce el resultado deseado.
        stabilized_motion_mesh_by_frame_index = np.reshape(
            vertex_stabilized_displacements_by_frame_index - vertex_unstabilized_displacements_by_frame_index,
            (num_frames, -1, 1, 2)
        )

        # Construir mapa desde el marco estabilizado al marco no estabilizado.
        # If (x_s, y_s) en el video estabilizado se toma de (x_u, y_u) en el no estabilizado
        # video, entonces
        # stabilized_y_x_to_unstabilized_x [y_s, x_s] = x_u,
        # stabilized_y_x_to_unstabilized_y [y_s, x_s] = y_u, y
        # Frame_stabilized_y_x_to_stabilized_x_y [y_s, x_s] = [x_u, y_u].
        # Tenga en cuenta el orden de coordenadas invertidas.Esta configuración nos permite indexar en el mapa al igual que
        # indexamos en la imagen.Cada punto [x_u, y_u] en la matriz es esperado
        # Ordene para que podamos aplicar fácilmente homografías a esos puntos.
        # Tenga en cuenta si los pasos posteriores no cambian el valor de una coordenada dada, entonces eso
        # La coordenada cae fuera de la imagen estabilizada (así que en la imagen de salida, esa imagen
        # debe llenarse con un color de borde).
        # Dado que los valores predeterminados de estas matrices caen fuera de la imagen no estabilizada, la reasigna
        # Complete esas coordenadas en la imagen estabilizada con el color del borde como se desee.
        frame_stabilized_y_x_to_unstabilized_x_template = np.full((frame_height, frame_width), frame_width + 1)
        frame_stabilized_y_x_to_unstabilized_y_template = np.full((frame_height, frame_width), frame_height + 1)
        frame_stabilized_y_x_to_stabilized_x_y_template = np.swapaxes(np.indices((frame_width, frame_height), dtype=np.float32), 0, 2)
        frame_stabilized_x_y_template = frame_stabilized_y_x_to_stabilized_x_y_template.reshape((-1, 1, 2))

        # Left_crop_x_by_frame_index [frame_index] contiene el valor X donde el borde izquierdo
        # donde frame_index se recortaría para producir una imagen rectangular;
        # right_crop_x_by_frame_index, top_crop_y_by_frame_index, y
        # bottom_crop_y_by_frame_index son análogas
        left_crop_x_by_frame_index = np.full(num_frames, 0)
        right_crop_x_by_frame_index = np.full(num_frames, frame_width - 1)
        top_crop_y_by_frame_index = np.full(num_frames, 0)
        bottom_crop_y_by_frame_index = np.full(num_frames, frame_height - 1)

        stabilized_frames = []
        with tqdm.trange(num_frames) as t:
            t.set_description('Warping frames')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]

                # Construir mapa desde el marco estabilizado al marco no estabilizado.
                # If (x_s, y_s) en el video estabilizado se toma de (x_u, y_u) en el no estabilizado
                # video, entonces
                # stabilized_y_x_to_unstabilized_x [y_s, x_s] = x_u,
                # stabilized_y_x_to_unstabilized_y [y_s, x_s] = y_u, y
                # Frame_stabilized_y_x_to_stabilized_x_y [y_s, x_s] = [x_u, y_u].
                # Tenga en cuenta el orden de coordenadas invertidas.Esta configuración nos permite indexar en el mapa al igual que
                # indexamos en la imagen.Cada punto [x_u, y_u] en la matriz es esperado
                # Ordene para que podamos aplicar fácilmente homografías a esos puntos.
                # Tenga en cuenta si los pasos posteriores no cambian el valor de una coordenada dada, entonces eso
                # La coordenada cae fuera de la imagen estabilizada (así que en la imagen de salida, esa imagen
                # debe llenarse con un color de borde).
                # Dado que los valores predeterminados de estas matrices caen fuera de la imagen no estabilizada, la reasigna
                # Complete esas coordenadas en la imagen estabilizada con el color del borde como se desee.
                frame_stabilized_y_x_to_unstabilized_x = np.copy(frame_stabilized_y_x_to_unstabilized_x_template)
                frame_stabilized_y_x_to_unstabilized_y = np.copy(frame_stabilized_y_x_to_unstabilized_y_template)
                frame_stabilized_x_y = np.copy(frame_stabilized_x_y_template)

                # Determine las coordenadas de los vértices de malla en el video estabilizado.
                # Los desplazamientos actuales están dados por VERTEX_UNSTABILED_DISPLACEMENTS, y
                # Los desplazamientos deseados están dados por Vertex_Stabilized_Displacements,
                # Entonces, agregar la diferencia de los dos transforma el marco como se desea.
                stabilized_vertex_x_y = unstabilized_vertex_x_y + stabilized_motion_mesh_by_frame_index[frame_index]

                row_col_to_stabilized_vertex_x_y = np.reshape(stabilized_vertex_x_y, (self.mesh_row_count + 1, self.mesh_col_count + 1, 2))
                # Mira cada cara de la malla.Ya que conocemos las coordenadas originales y transformadas
                # De sus cuatro vértices, podemos construir una homografía para completar los píxeles restantes
                # Todos paralelizan
                for cell_top_left_row in range(self.mesh_row_count):
                    for cell_top_left_col in range(self.mesh_col_count):

                        # Construya una máscara que represente la célula estabilizada.
                        # Dado que conocemos los límites de la célula antes y después de la estabilización, podemos
                        # Construya una homografía que represente la urdimbre de esta célula y luego apliquela a
                        # La celda no estabilizada (que es solo un rectángulo) para construir la estabilizada
                        # celúla.
                        unstabilized_cell_bounds = row_col_to_unstabilized_vertex_x_y[cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        stabilized_cell_bounds = row_col_to_stabilized_vertex_x_y[cell_top_left_row:cell_top_left_row+2, cell_top_left_col:cell_top_left_col+2].reshape(-1, 2)
                        unstabilized_to_stabilized_homography, _ = cv2.findHomography(unstabilized_cell_bounds, stabilized_cell_bounds)
                        stabilized_to_unstabilized_homography, _ = cv2.findHomography(stabilized_cell_bounds, unstabilized_cell_bounds)

                        unstabilized_cell_x_bounds, unstabilized_cell_y_bounds = np.transpose(unstabilized_cell_bounds)
                        unstabilized_cell_left_x = math.floor(np.min(unstabilized_cell_x_bounds))
                        unstabilized_cell_right_x = math.ceil(np.max(unstabilized_cell_x_bounds))
                        unstabilized_cell_top_y = math.floor(np.min(unstabilized_cell_y_bounds))
                        unstabilized_cell_bottom_y = math.ceil(np.max(unstabilized_cell_y_bounds))

                        unstabilized_cell_mask = np.zeros((frame_height, frame_width))
                        unstabilized_cell_mask[unstabilized_cell_top_y:unstabilized_cell_bottom_y+1, unstabilized_cell_left_x:unstabilized_cell_right_x+1] = 255
                        stabilized_cell_mask = cv2.warpPerspective(unstabilized_cell_mask, unstabilized_to_stabilized_homography, (frame_width, frame_height))

                        cell_unstabilized_x_y = cv2.perspectiveTransform(frame_stabilized_x_y, stabilized_to_unstabilized_homography)
                        cell_stabilized_y_x_to_unstabilized_x_y = cell_unstabilized_x_y.reshape((frame_height, frame_width, 2))
                        cell_stabilized_y_x_to_unstabilized_x, cell_stabilized_y_x_to_unstabilized_y = np.moveaxis(cell_stabilized_y_x_to_unstabilized_x_y, 2, 0)

                        # Actualizar el mapa general estabilizado a no estabilizado, aplicando esta celda
                        # Transformación solo para aquellos píxeles que en realidad son parte de esta celda
                        frame_stabilized_y_x_to_unstabilized_x = np.where(stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_x, frame_stabilized_y_x_to_unstabilized_x)
                        frame_stabilized_y_x_to_unstabilized_y = np.where(stabilized_cell_mask, cell_stabilized_y_x_to_unstabilized_y, frame_stabilized_y_x_to_unstabilized_y)

                # Este método se utiliza para mapear los píxeles de una imagen a nuevas
                #UbicacionesBasadasEnMapasDeCoordenadas
                stabilized_frame = cv2.remap(
                    unstabilized_frame,
                    frame_stabilized_y_x_to_unstabilized_x.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    frame_stabilized_y_x_to_unstabilized_y.reshape((frame_height, frame_width, 1)).astype(np.float32),
                    cv2.INTER_LINEAR,
                    borderValue=self.color_outside_image_area_bgr
                )

                # recortar el marco
                # Edge izquierdo: el X_S estabilizado máximo que corresponde al no estabilizado
                #XU =0

                stabilized_image_x_matching_unstabilized_left_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_x - 0) < 1)[1]
                if stabilized_image_x_matching_unstabilized_left_edge.size > 0:
                    left_crop_x_by_frame_index[frame_index] = np.max(stabilized_image_x_matching_unstabilized_left_edge)

                # Edge derecho: el X_S estabilizado mínimo que corresponde al estabilizado
                #XU =AnchoDeMarco1

                stabilized_image_x_matching_unstabilized_right_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_x - (frame_width - 1)) < 1)[1]
                if stabilized_image_x_matching_unstabilized_right_edge.size > 0:
                    right_crop_x_by_frame_index[frame_index] = np.min(stabilized_image_x_matching_unstabilized_right_edge)

                # borde superior: el y_s estabilizado máximo que corresponde al no estabilizado
                #YU =0

                stabilized_image_y_matching_unstabilized_top_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_y - 0) < 1)[0]
                if stabilized_image_y_matching_unstabilized_top_edge.size > 0:
                    top_crop_y_by_frame_index[frame_index] = np.max(stabilized_image_y_matching_unstabilized_top_edge)

                # Borde inferior: el y_ estabilizado mínimo que corresponde al no estabilizado
                #YU =AlturaDelMarco1

                stabilized_image_y_matching_unstabilized_bottom_edge = np.where(np.abs(frame_stabilized_y_x_to_unstabilized_y - (frame_height - 1)) < 1)[0]
                if stabilized_image_y_matching_unstabilized_bottom_edge.size > 0:
                    bottom_crop_y_by_frame_index[frame_index] = np.min(stabilized_image_y_matching_unstabilized_bottom_edge)

                stabilized_frames.append(stabilized_frame)

        # La cosecha de video final es la que recortaría adecuadamente cada cuadro
        left_crop_x = np.max(left_crop_x_by_frame_index)
        right_crop_x = np.min(right_crop_x_by_frame_index)
        top_crop_y = np.max(top_crop_y_by_frame_index)
        bottom_crop_y = np.min(bottom_crop_y_by_frame_index)

        return (stabilized_frames, (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y))


    def _crop_frames(self, uncropped_frames, crop_boundaries):
        '''
        Return copies of the given frames that have been cropped according to the given crop
        boundaries.

        Input:

        * uncropped_frames: A list of the frames to crop, each represented as a NumPy array.
        * crop_boundaries: A tuple of the form
            (left_crop_x, top_crop_y, right_crop_x, bottom_crop_y)
            representing the x- and y-boundaries (all inclusive) of the cropped video.

        Output:

        * cropped_frames: A list of the frames cropped according to the crop boundaries.

        '''

        frame_height, frame_width = uncropped_frames[0].shape[:2]
        left_crop_x, top_crop_y, right_crop_x, bottom_crop_y = crop_boundaries

        # Hay dos formas de ampliar la imagen: aumentar su ancho para llenar el ancho original,
        # Escalar la altura adecuadamente, o aumentar su altura para llenar la altura original,
        # Escalar el ancho apropiadamente.Al menos una de estas opciones dará como resultado la imagen
        # Completar completamente el marco.
        uncropped_aspect_ratio = frame_width / frame_height
        cropped_aspect_ratio = (right_crop_x + 1 - left_crop_x) / (bottom_crop_y + 1 - top_crop_y)

        if cropped_aspect_ratio >= uncropped_aspect_ratio:
            # La imagen recortada es proporcionalmente más ancha que la original, por lo que para llenar completamente el
            # marco, debe escalarse para que su altura coincida con la altura del marco
            uncropped_to_cropped_scale_factor = frame_height / (bottom_crop_y + 1 - top_crop_y)
        else:
            # La imagen recortada es proporcionalmente más alta que la original, por lo que para llenar por completo
            # El marco, debe escalarse para que su ancho coincida con el ancho del marco
            uncropped_to_cropped_scale_factor = frame_width / (right_crop_x + 1 - left_crop_x)

        cropped_frames = []
        for uncropped_frame in uncropped_frames:
            cropped_frames.append(cv2.resize(
                uncropped_frame[top_crop_y:bottom_crop_y+1, left_crop_x:right_crop_x+1],
                (frame_width, frame_height),
                fx=uncropped_to_cropped_scale_factor,
                fy=uncropped_to_cropped_scale_factor
            ))

        return cropped_frames


    def _compute_cropping_ratio_and_distortion_score(self, num_frames, unstabilized_frames, cropped_frames):
        '''
        Helper function for stabilize.

        Compute the cropping ratio and distortion score for the given stabilization using the
        definitions of these metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * stabilized_frames: A list of the stabilized frames, each represented as a NumPy array.

        Output:

        A tuple of the following items in order.

        * cropping_ratio: The cropping ratio of the stabilized video. Per the original paper, the
            cropping ratio of each frame is the scale component of its unstabilized-to-cropped
            homography, and the cropping ratio of the overall video is the average of the frames'
            cropping ratios.
        * distortion_score: The distortion score of the stabilized video. Per the original paper,
            the distortion score of each frame is ratio of the two largest eigenvalues of the
            affine part of its unstabilized-to-cropped homography, and the distortion score of the
            overall video is the greatest of its frames' distortion scores.
        '''

        cropping_ratios = np.empty((num_frames), dtype=np.float32)
        distortion_scores = np.empty((num_frames), dtype=np.float32)

        with tqdm.trange(num_frames) as t:
            t.set_description('Computing cropping ratio and distortion score')
            for frame_index in t:
                unstabilized_frame = unstabilized_frames[frame_index]
                cropped_frame = cropped_frames[frame_index]
                _, _, unstabilized_to_cropped_homography = self._get_matched_features_and_homography(
                    unstabilized_frame, cropped_frame
                )

                # El componente de escala tiene X-Component Cropped_to_unstabilized_homography [0] [0]
                # e componente y componente cropped_to_unstabilized_homography [1] [1],
                # Entonces, la fracción del video ampliado que realmente se ajusta en el marco es
                # 1 / (Cropped_to_unstabilized_homography [0] [0] * Cropped_to_unstabilized_homography [1] [1])
                cropping_ratio = 1 / (unstabilized_to_cropped_homography[0][0] * unstabilized_to_cropped_homography[1][1])
                cropping_ratios[frame_index] = cropping_ratio

                affine_component = np.copy(unstabilized_to_cropped_homography)
                affine_component[2] = [0, 0, 1]
                eigenvalue_magnitudes = np.sort(np.abs(np.linalg.eigvals(affine_component)))
                distortion_score = eigenvalue_magnitudes[-2] / eigenvalue_magnitudes[-1]
                distortion_scores[frame_index] = distortion_score

        return (np.mean(cropping_ratios), np.min(distortion_scores))



    def _compute_stability_score(self, num_frames, vertex_stabilized_displacements_by_frame_index):
        '''
        Helper function for stabilize.

        Compute the stability score for the given stabilization using the definitions of these
        metrics in the original paper.

        Input:

        * num_frames: The number of frames in the video.
        * vertex_stabilized_displacements_by_frame_index: A NumPy array containing the
            stabilized displacements of each vertex in the  mesh, as generated by
            _get_stabilized_vertex_displacements.

        Output:

        * stability_score: The stability score of the stabilized video. Per the original paper, the
            stability score for each vertex is derived from the representation of its vertex profile
            (vector of velocities) in the frequency domain. Specifically, it is the fraction of the
            representation's total energy that is contained within its second to sixth lowest
            frequencies. The stability score of the overall video is the average of the vertices'
            average x- and y-stability scores.
        '''

        vertex_stabilized_x_dispacements_by_row_and_col, vertex_stabilized_y_dispacements_by_row_and_col = np.swapaxes(vertex_stabilized_displacements_by_frame_index, 0, 3)
        vertex_x_profiles_by_row_and_col = np.diff(vertex_stabilized_x_dispacements_by_row_and_col)
        vertex_y_profiles_by_row_and_col = np.diff(vertex_stabilized_y_dispacements_by_row_and_col)

        vertex_x_freq_energies_by_row_and_col = np.square(np.abs(np.fft.fft(vertex_x_profiles_by_row_and_col)))
        vertex_y_freq_energies_by_row_and_col = np.square(np.abs(np.fft.fft(vertex_y_profiles_by_row_and_col)))

        vertex_x_total_freq_energy_by_row_and_col = np.sum(vertex_x_freq_energies_by_row_and_col, axis=2)
        vertex_y_total_freq_energy_by_row_and_col = np.sum(vertex_y_freq_energies_by_row_and_col, axis=2)

        vertex_x_low_freq_energy_by_row_and_col = np.sum(vertex_x_freq_energies_by_row_and_col[:, :, 1:6], axis=2)
        vertex_y_low_freq_energy_by_row_and_col = np.sum(vertex_y_freq_energies_by_row_and_col[:, :, 1:6], axis=2)

        x_stability_scores_by_row_and_col = vertex_x_low_freq_energy_by_row_and_col / vertex_x_total_freq_energy_by_row_and_col
        y_stability_scores_by_row_and_col = vertex_y_low_freq_energy_by_row_and_col / vertex_y_total_freq_energy_by_row_and_col

        x_stability_score = np.mean(x_stability_scores_by_row_and_col)
        y_stability_score = np.mean(y_stability_scores_by_row_and_col)

        return (x_stability_score + y_stability_score) / 2.0


    def _display_unstablilized_and_cropped_video_loop(self, num_frames, frames_per_second, unstabilized_frames, cropped_frames):
        '''
        Helper function for stabilize.

        Display a loop of the unstabilized and cropped, stabilized videos.

        Input:

        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * unstabilized_frames: A list of the unstabilized frames, each represented as a NumPy array.
        * cropped_frames: A list of the cropped, stabilized frames, each represented as a NumPy
            array.

        Output:

        (The unstabilized and cropped, stabilized videos loop in a new window. Pressing the Q key
        closes the window.)
        '''

        milliseconds_per_frame = int(1000/frames_per_second)
        while True:
            for i in range(num_frames):
                cv2.imshow('unstabilized and stabilized video', np.vstack((unstabilized_frames[i], cropped_frames[i])))
                if cv2.waitKey(milliseconds_per_frame) & 0xFF == ord('q'):
                    return


    def _write_stabilized_video(self, output_path, num_frames, frames_per_second, codec, stabilized_frames):
        '''
        Helper method for stabilize.
        Write the given stabilized frames as a video to the given path.

        Input:
        * output_path: The path where the stabilized version of the video should be placed.
        * num_frames: The number of frames in the video.
        * frames_per_second: The video framerate in frames per second.
        * codec: The video codec.
        * stabilized_frames: A list of the frames in the stabilized video, each represented as a
            NumPy array.

        Output:

        (The video is saved to output_path.)
        '''

        # adaptado de https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/
        frame_height, frame_width = stabilized_frames[0].shape[:2]
        video = cv2.VideoWriter(
            output_path,
            codec,
            frames_per_second,
            (frame_width, frame_height)
        )

        with tqdm.trange(num_frames) as t:
            t.set_description(f'Writing stabilized video to <{output_path}>')
            for frame_index in t:
                video.write(stabilized_frames[frame_index])

        video.release()


def main():
    # TODO get video path from command line args

    input_path = 'videos/video-11/VID_20240825_140454cps_.mp4' if len(sys.argv) < 2 else sys.argv[1]
    output_path = 'videos/video-11/stabilized-method-original.m4v' if len(sys.argv) < 3 else sys.argv[2]
    stabilizer = Stabilizer(visualize=True)
    cropping_ratio, distortion_score, stability_score = stabilizer.stabilize(
        input_path, output_path,
        adaptive_weights_definition=Stabilizer.ADAPTIVE_WEIGHTS_DEFINITION_ORIGINAL,
    )
    print('cropping ratio:', cropping_ratio)
    print('distortion score:', distortion_score)
    print('stability score:', stability_score)


if __name__ == '__main__':
    main()