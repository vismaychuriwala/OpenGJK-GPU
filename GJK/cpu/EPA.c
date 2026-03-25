
//                           _____      _ _  __ //
//                          / ____|    | | |/ / //
//    ___  _ __   ___ _ __ | |  __     | | ' / //
//   / _ \| '_ \ / _ \ '_ \| | |_ |_   | |  < //
//  | (_) | |_) |  __/ | | | |__| | |__| | . \ //
//   \___/| .__/ \___|_| |_|\_____|\____/|_|\_\ //
//        | | //
//        |_| //
//                                                                                //
// Copyright 2022 Mattia Montanari, University of Oxford //
//                                                                               //
// This program is free software: you can redistribute it and/or modify it under
// // the terms of the GNU General Public License as published by the Free
// Software  // Foundation, either version 3 of the License. You should have
// received a copy   // of the GNU General Public License along with this
// program. If not, visit       //
//                                                                                //
//     https://www.gnu.org/licenses/ //
//                                                                                //
// This program is distributed in the hope that it will be useful, but WITHOUT
// // ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS  // FOR A PARTICULAR PURPOSE. See GNU General Public License for
// details.          //

/**
 * @file openGJK.h
 * @author Marcus Headlund and Vismay Churiwala
 * @date 1 Jan 2026
 * @brief Main interface of the EPA algorithm.
 *
 */

#include "EPA.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <math.h>

// Maximum number of faces in the EPA polytope
#define MAX_EPA_FACES 128
#define MAX_EPA_VERTICES (MAX_EPA_FACES + 4)

#define eps_rel22 ((gkFloat)(gkEpsilon) * (gkFloat)1e4)
#define eps_tot22 ((gkFloat)(gkEpsilon) * (gkFloat)1e2)

#define norm2(a) (a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
#define getCoord(body, index, component) body->coord[(index) * 3 + (component)]
#define dotProduct(a, b) (a[0] * b[0] + a[1] * b[1] + a[2] * b[2])

// Entry point to EPA implementation.

// Face structure for EPA polytope
// Each face is a triangle with 3 vertex indices
typedef struct {
  int v[3];           // Vertex indices in the polytope
  int v_idx[3][2];    // Original vertex indices from original polytopes for witness computation [vertex][body]
  gkFloat normal[3];  // Face normal (pointing outward from origin)
  gkFloat distance;   // Distance from origin to face plane
  bool valid;         // Whether this face is still valid (not removed)
} EPAFace;

// Polytope structure for EPA
typedef struct {
  gkFloat vertices[MAX_EPA_FACES + 4][3];  // Vertex coordinates in the Minkowski difference
  int vertex_indices[MAX_EPA_FACES + 4][2]; // Original vertex indices [vertex][body]
  int num_vertices;
  EPAFace faces[MAX_EPA_FACES];
  int max_face_index; // Highest face index in use (for iteration bounds)
} EPAPolytope;


// Structure for horizon edge collection
typedef struct {
  int v1, v2;  // Vertex indices in polytope
  int v_idx1[2], v_idx2[2];  // Original vertex indices for witness computation
  bool valid;
} EPAEdge;

inline static void crossProduct(const gkFloat* a,
  const gkFloat* b,
  gkFloat* c) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

// Compute face normal and distance of face from origin.
// Winding is already fixed at face creation time, so the cross product
// direction is trusted directly — no centroid-based orientation check needed.
inline static void compute_face_normal_distance(EPAPolytope* poly, int face_idx) {
  EPAFace* face = &poly->faces[face_idx];

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat* v1 = poly->vertices[face->v[1]];
  gkFloat* v2 = poly->vertices[face->v[2]];

  gkFloat e0[3], e1[3];
  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
  }

  crossProduct(e0, e1, face->normal);
  gkFloat norm_sq = norm2(face->normal);

  if (norm_sq > gkEpsilon * gkEpsilon) {
    gkFloat norm = gkSqrt(norm_sq);
    for (int i = 0; i < 3; i++) {
      face->normal[i] /= norm;
    }

    face->distance = dotProduct(face->normal, v0);

    // Safety: origin should be inside polytope so distance must be positive.
    // If negative, the winding was wrong — flip to recover.
    if (face->distance < 0) {
      for (int i = 0; i < 3; i++) {
        face->normal[i] = -face->normal[i];
      }
      face->distance = -face->distance;
    }
  }
  else {
    face->valid = false;
    face->distance = (gkFloat)1e10;
  }
}

// Check if a face is visible from a point (point is on positive side of face) Needed to determine which faces to restructure when vertex is added
inline static bool is_face_visible(EPAPolytope* poly, int face_idx, const gkFloat* point) {
  EPAFace* face = &poly->faces[face_idx];
  if (!face->valid) return false;

  gkFloat* v0 = poly->vertices[face->v[0]];
  gkFloat diff[3];
  for (int i = 0; i < 3; i++) diff[i] = point[i] - v0[i];
  return dotProduct(face->normal, diff) > gkEpsilon;
}


// Initialize EPA polytope from GJK simplex (should be a tetrahedron)
inline static void init_epa_polytope(EPAPolytope* poly, const gkSimplex* simplex, gkFloat* centroid) {
  memset(poly->faces, 0, sizeof(poly->faces));

  // Copy vertices from simplex
  poly->num_vertices = 4;
  for (int i = 0; i < 4; i++) {
    
    for (int j = 0; j < 3; j++) {
      poly->vertices[i][j] = simplex->vrtx[i][j];
    }
    poly->vertex_indices[i][0] = simplex->vrtx_idx[i][0];
    poly->vertex_indices[i][1] = simplex->vrtx_idx[i][1];
  }

  // Compute centroid of the tetrahedron
  centroid[0] = centroid[1] = centroid[2] = 0.0f;

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 3; j++) {
      centroid[j] += poly->vertices[i][j] * 0.25f;
    }
  }

  // Create 4 faces of tetrahedron
  // set up the faces and then fix winding based on normal direction
  // Face 0: vertices 0, 1, 2
  poly->faces[0].v[0] = 0;
  poly->faces[0].v[1] = 1;
  poly->faces[0].v[2] = 2;
  poly->faces[0].valid = true;

  // Face 1: vertices 0, 3, 1
  poly->faces[1].v[0] = 0;
  poly->faces[1].v[1] = 3;
  poly->faces[1].v[2] = 1;
  poly->faces[1].valid = true;

  // Face 2: vertices 0, 2, 3
  poly->faces[2].v[0] = 0;
  poly->faces[2].v[1] = 2;
  poly->faces[2].v[2] = 3;
  poly->faces[2].valid = true;

  // Face 3: vertices 1, 3, 2
  poly->faces[3].v[0] = 1;
  poly->faces[3].v[1] = 3;
  poly->faces[3].v[2] = 2;
  poly->faces[3].valid = true;

  // Copy vertex indices for witness point computation
  
  for (int f = 0; f < 4; f++) {
    for (int v = 0; v < 3; v++) {
      int vi = poly->faces[f].v[v];
      poly->faces[f].v_idx[v][0] = simplex->vrtx_idx[vi][0];
      poly->faces[f].v_idx[v][1] = simplex->vrtx_idx[vi][1];
    }
  }

  // Compute normals and fix winding
  for (int f = 0; f < 4; f++) {
    gkFloat* v0 = poly->vertices[poly->faces[f].v[0]];
    gkFloat* v1 = poly->vertices[poly->faces[f].v[1]];
    gkFloat* v2 = poly->vertices[poly->faces[f].v[2]];

    gkFloat e0[3], e1[3], normal[3];
    
    for (int i = 0; i < 3; i++) {
      e0[i] = v1[i] - v0[i];
      e1[i] = v2[i] - v0[i];
    }
    crossProduct(e0, e1, normal);

    // If normal points toward centroid need to flip the winding
    gkFloat to_centroid[3];
    for (int i = 0; i < 3; i++) to_centroid[i] = centroid[i] - v0[i];
    if (dotProduct(normal, to_centroid) > 0) {
      int tmp = poly->faces[f].v[1];
      poly->faces[f].v[1] = poly->faces[f].v[2];
      poly->faces[f].v[2] = tmp;

      int tmp_idx0 = poly->faces[f].v_idx[1][0];
      int tmp_idx1 = poly->faces[f].v_idx[1][1];
      poly->faces[f].v_idx[1][0] = poly->faces[f].v_idx[2][0];
      poly->faces[f].v_idx[1][1] = poly->faces[f].v_idx[2][1];
      poly->faces[f].v_idx[2][0] = tmp_idx0;
      poly->faces[f].v_idx[2][1] = tmp_idx1;
    }
  }

  poly->max_face_index = 4;
}

// barycentric coordinate compute closest point on triangle to origin
inline static void compute_barycentric_origin(
  const gkFloat* v0, const gkFloat* v1, const gkFloat* v2,
  gkFloat* a0, gkFloat* a1, gkFloat* a2) {

  // Compute vectors
  gkFloat e0[3], e1[3];

  for (int i = 0; i < 3; i++) {
    e0[i] = v1[i] - v0[i];
    e1[i] = v2[i] - v0[i];
  }

  // Compute dot products for barycentric coords
  gkFloat d00 = dotProduct(e0, e0);
  gkFloat d01 = dotProduct(e0, e1);
  gkFloat d11 = dotProduct(e1, e1);
  gkFloat d20 = -dotProduct(v0, e0);
  gkFloat d21 = -dotProduct(v0, e1);

  gkFloat denom = d00 * d11 - d01 * d01;

  if (gkFabs(denom) < gkEpsilon) {
    // Degenerate
    *a0 = *a1 = *a2 = (gkFloat)1.0 / (gkFloat)3.0;
    return;
  }

  gkFloat inv_denom = (gkFloat)1.0 / denom;
  gkFloat u = (d11 * d20 - d01 * d21) * inv_denom;
  gkFloat v = (d00 * d21 - d01 * d20) * inv_denom;
  gkFloat w = (gkFloat)1.0 - u - v;

  // Clamp to triangle
  if (w < 0) {
    // Origin projects outside edge v1-v2
    // Project onto edge v1-v2
    gkFloat e12[3];
    for (int i = 0; i < 3; i++) e12[i] = v2[i] - v1[i];
    gkFloat t = -dotProduct(v1, e12) / dotProduct(e12, e12);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
    *a0 = 0;
    *a1 = (gkFloat)1.0 - t;
    *a2 = t;
  }
  else if (u < 0) {
    // Origin projects outside edge v0-v2
    gkFloat t = -dotProduct(v0, e1) / dotProduct(e1, e1);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
    *a0 = (gkFloat)1.0 - t;
    *a1 = 0;
    *a2 = t;
  }
  else if (v < 0) {
    // Origin projects outside edge v0-v1
    gkFloat t = -dotProduct(v0, e0) / dotProduct(e0, e0);
    t = gkFmax((gkFloat)0.0, gkFmin((gkFloat)1.0, t));
    *a0 = (gkFloat)1.0 - t;
    *a1 = t;
    *a2 = 0;
  }
  else {
    // Inside triangle
    *a0 = w;
    *a1 = u;
    *a2 = v;
  }
}

// Support function for EPA basicallly GJK one but only care about minkowski difference point
inline static void support_epa(const gkPolytope* body1, const gkPolytope* body2,
  const gkFloat* direction, gkFloat* result, int* result_idx) {

  gkFloat local_max1 = -1e10f;
  gkFloat local_max2 = -1e10f;
  int local_best1 = -1;
  int local_best2 = -1;

  // Search body1
  for (int i = 0; i < body1->numpoints; i++) {
    gkFloat s = getCoord(body1, i, 0) * direction[0]
              + getCoord(body1, i, 1) * direction[1]
              + getCoord(body1, i, 2) * direction[2];
    if (s > local_max1) {
      local_max1 = s;
      local_best1 = i;
    }
  }

  // Search body2 in opposite direction
  for (int i = 0; i < body2->numpoints; i++) {
    gkFloat s = getCoord(body2, i, 0) * direction[0]
              + getCoord(body2, i, 1) * direction[1]
              + getCoord(body2, i, 2) * direction[2];
    if (-s > local_max2) {
      local_max2 = -s;
      local_best2 = i;
    }
  }
  // Compute Minkowski difference point
  if (local_best1 >= 0 && local_best2 >= 0) {
    for (int i = 0; i < 3; i++) {
      result[i] = getCoord(body1, local_best1, i) - getCoord(body2, local_best2, i);
    }
    result_idx[0] = local_best1;
    result_idx[1] = local_best2;
  }
}

//*******************************************************************************************
// EPA Implementation
//*******************************************************************************************

static void set_contact_normal(const gkFloat* w1, const gkFloat* w2, gkFloat* contact_normal) {
  gkFloat d[3] = { w2[0] - w1[0], w2[1] - w1[1], w2[2] - w1[2] };
  gkFloat n = gkSqrt(norm2(d));
  if (n > gkEpsilon) {
    contact_normal[0] = d[0] / n;
    contact_normal[1] = d[1] / n;
    contact_normal[2] = d[2] / n;
  } else {
    contact_normal[0] = 1.0f; contact_normal[1] = 0.0f; contact_normal[2] = 0.0f;
  }
}

  void computeCollisionInformation(
  const gkPolytope* bd1,
  const gkPolytope* bd2,
  gkSimplex* simplex,
  gkFloat* distance,
  gkFloat contact_normal[3]) {

  // if distance isn't 0 didn't detect collision - skip EPA
  if (*distance > gkEpsilon) {
    set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
    return;
  }

  // If GJK returned a degenerate simplex, rebuild it properly for EPA
  if (simplex->nvrtx != 4) {
    // Need to get it up to 4 vertices
    if (simplex->nvrtx == 1) {
      // Grow simplex from a single point: fire a support in some direction.
      // We use current simplex point for new direction for the
      // support; if this does not produce a new point, treat penetration as 0.
        gkFloat new_vertex[3];
        int new_vertex_idx[2];
        const gkFloat eps_sq = gkEpsilon * gkEpsilon;

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, simplex->vrtx[0], new_vertex, new_vertex_idx);

        // Check if this is a new point relative to the existing simplex vertex.
        bool is_new = true;
        gkFloat dx = new_vertex[0] - simplex->vrtx[0][0];
        gkFloat dy = new_vertex[1] - simplex->vrtx[0][1];
        gkFloat dz = new_vertex[2] - simplex->vrtx[0][2];
        gkFloat d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < eps_sq) {
          is_new = false;
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 2;
        }
        else {
          // No new support point means penetration depth effectively zero.
          *distance = 0.0f;
          for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
            simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
          }
          set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
          return;
        }
    }
    if (simplex->nvrtx == 2) {
      // Grow simplex from an edge: fire a support in a direction perpendicular
      // to the edge. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;

    gkFloat edge[3];

    for (int c = 0; c < 3; ++c) {
        edge[c] = simplex->vrtx[1][c] - simplex->vrtx[0][c];
    }

    // Build a perpindicular
    gkFloat axis[3] = { 1.0f, 0.0f, 0.0f };
    gkFloat edge_norm = gkSqrt(norm2(edge));
    if (edge_norm > gkEpsilon && gkFabs(edge[0]) > 0.9f * edge_norm) {
        axis[0] = 0.0f; axis[1] = 1.0f; axis[2] = 0.0f;
    }

    // dir = edge x axis
    crossProduct(edge, axis, dir);
    gkFloat nrm2 = norm2(dir);
    if (nrm2 < gkEpsilon) {
        // Fallback axis
        axis[0] = 0.0f; axis[1] = 0.0f; axis[2] = 1.0f;
        crossProduct(edge, axis, dir);
    }

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
        // Check if this is a new point relative to both existing simplex vertices.
        bool is_new = true;
        for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 3;
        }
        else {
          // No new support point means penetration depth effectively zero.
          *distance = 0.0f;
          for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
            simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
          }
          set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
          return;
        }
    }
    if (simplex->nvrtx == 3) {
      // Grow simplex from a triangle: fire a support in the direction of the
      // triangle normal. If this does not produce a new point, treat penetration as 0.
      gkFloat dir[3];
      gkFloat new_vertex[3];
      int new_vertex_idx[2];
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;

        gkFloat e0[3], e1[3];

        for (int c = 0; c < 3; ++c) {
            e0[c] = simplex->vrtx[1][c] - simplex->vrtx[0][c];
            e1[c] = simplex->vrtx[2][c] - simplex->vrtx[0][c];
        }
        // dir = e0 x e1 (normal to the triangle)
        crossProduct(e0, e1, dir);

      // Parallel EPA support in that direction.
      support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
        // Check if this is a new point relative to all three existing simplex vertices.
        bool is_new = true;
        for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
          gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
          gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
          gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
          gkFloat d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < eps_sq) {
            is_new = false;
            break;
          }
        }

        if (is_new) {
          int idx = simplex->nvrtx;
          
          for (int c = 0; c < 3; ++c) {
            simplex->vrtx[idx][c] = new_vertex[c];
          }
          simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
          simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
          simplex->nvrtx = 4;
        }
        else {
          // Try opposite direction
          dir[0] = -dir[0];
          dir[1] = -dir[1];
          dir[2] = -dir[2];
        }

      // If first direction didn't work, try opposite
      if (simplex->nvrtx == 3) {
        support_epa(bd1, bd2, dir, new_vertex, new_vertex_idx);
          bool is_new = true;
          for (int vtx = 0; vtx < simplex->nvrtx; ++vtx) {
            gkFloat dx = new_vertex[0] - simplex->vrtx[vtx][0];
            gkFloat dy = new_vertex[1] - simplex->vrtx[vtx][1];
            gkFloat dz = new_vertex[2] - simplex->vrtx[vtx][2];
            gkFloat d2 = dx * dx + dy * dy + dz * dz;
            if (d2 < eps_sq) {
              is_new = false;
              break;
            }
          }

          if (is_new) {
            int idx = simplex->nvrtx;
            
            for (int c = 0; c < 3; ++c) {
              simplex->vrtx[idx][c] = new_vertex[c];
            }
            simplex->vrtx_idx[idx][0] = new_vertex_idx[0];
            simplex->vrtx_idx[idx][1] = new_vertex_idx[1];
            simplex->nvrtx = 4;
          }
          else {
            *distance = 0.0f;
            for (int c = 0; c < 3; ++c) {
              simplex->witnesses[0][c] = getCoord(bd1, new_vertex_idx[0], c);
              simplex->witnesses[1][c] = getCoord(bd2, new_vertex_idx[1], c);
            }
            set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
            return;
          }
      }
    }

    // If we still don't have 4 vertices, abort
    if (simplex->nvrtx != 4) {
        *distance = 0.0f;
        // Set witness points from best available simplex vertex
        int best = simplex->nvrtx > 0 ? simplex->nvrtx - 1 : 0;
        for (int c = 0; c < 3; ++c) {
            simplex->witnesses[0][c] = getCoord(bd1, simplex->vrtx_idx[best][0], c);
            simplex->witnesses[1][c] = getCoord(bd2, simplex->vrtx_idx[best][1], c);
        }
        set_contact_normal(simplex->witnesses[0], simplex->witnesses[1], contact_normal);
        return;
    }
  }

  // On to actual EPA alg with a valid tetrahedron simplex
  // Initialize EPA polytope from simplex
  EPAPolytope poly;
  gkFloat centroid[3];
    init_epa_polytope(&poly, simplex, centroid);

  // EPA iteration parameters
  const int max_iterations = 64;
  const gkFloat tolerance = eps_tot22;
  int iteration = 0;

  // Main EPA loop
  while (iteration < max_iterations && poly.num_vertices < MAX_EPA_VERTICES - 1) {
    iteration++;
    // Recompute normals & distances for assigned faces
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (poly.faces[i].valid) {
        compute_face_normal_distance(&poly, i);
      }
    }

    // parallel reduction to find closest face
    // finds the closest face in the range
    int closest_face = -1;
    gkFloat closest_distance = 1e10f;

    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      if (poly.faces[i].distance >= 0.0f && poly.faces[i].distance < closest_distance) {
        closest_distance = poly.faces[i].distance;
        closest_face = i;
      }
    }


    if (closest_face < 0) {
      break;
    }

    EPAFace* closest = &poly.faces[closest_face];

    // Get support point in direction of closest face normal
    gkFloat new_vertex[3];
    int new_vertex_idx[2];
    support_epa(bd1, bd2, poly.faces[closest_face].normal, new_vertex, new_vertex_idx);

    // Check termination condition: if distance to new vertex along normal is not more than tolerance further than closest face
    gkFloat dist_to_new = dotProduct(poly.faces[closest_face].normal, new_vertex);
    gkFloat improvement = dist_to_new - closest_distance;

    if (improvement < tolerance) {
      // Converged, compute witness points with bary coords
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
      break;
    }

    /// Check if new vertex is duplicate
    bool is_duplicate = false;
      const gkFloat eps_sq = gkEpsilon * gkEpsilon;
      for (int i = 0; i < poly.num_vertices; i++) {
        gkFloat dx = new_vertex[0] - poly.vertices[i][0];
        gkFloat dy = new_vertex[1] - poly.vertices[i][1];
        gkFloat dz = new_vertex[2] - poly.vertices[i][2];
        if (dx * dx + dy * dy + dz * dz < eps_sq) {
          is_duplicate = true;
          break;
        }
      }

    if (is_duplicate) {
      // Can't make progress, use current best
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
      break;
    }

    // Add new vertex to polytope
    int new_vertex_id = poly.num_vertices;
    for (int i = 0; i < 3; i++) {
      poly.vertices[new_vertex_id][i] = new_vertex[i];
    }
    poly.vertex_indices[new_vertex_id][0] = new_vertex_idx[0];
    poly.vertex_indices[new_vertex_id][1] = new_vertex_idx[1];
    poly.num_vertices++;

    // Update centroid incrementally (running mean)
    gkFloat inv_n = (gkFloat)1.0 / (gkFloat)poly.num_vertices;
    for (int i = 0; i < 3; i++) {
      centroid[i] += (new_vertex[i] - centroid[i]) * inv_n;
    }
    // Find horizon edges: collect edges from faces being removed this iteration
    // only, then mark them invalid. Collecting from ALL invalid faces (including
    // ones from previous iterations) would pull in stale interior edges.
      EPAEdge edges[MAX_EPA_FACES * 3];
      int num_edges = 0;

      for (int f = 0; f < poly.max_face_index; f++) {
        if (!poly.faces[f].valid) continue;
        if (!is_face_visible(&poly, f, new_vertex)) continue;

        // Collect edges before invalidating the face
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[0];
          edges[num_edges].v2 = poly.faces[f].v[1];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[0][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[0][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[1][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[1][1];
          edges[num_edges].valid = true;
          num_edges++;
        }
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[1];
          edges[num_edges].v2 = poly.faces[f].v[2];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[1][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[1][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[2][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[2][1];
          edges[num_edges].valid = true;
          num_edges++;
        }
        if (num_edges < MAX_EPA_FACES * 3) {
          edges[num_edges].v1 = poly.faces[f].v[2];
          edges[num_edges].v2 = poly.faces[f].v[0];
          edges[num_edges].v_idx1[0] = poly.faces[f].v_idx[2][0];
          edges[num_edges].v_idx1[1] = poly.faces[f].v_idx[2][1];
          edges[num_edges].v_idx2[0] = poly.faces[f].v_idx[0][0];
          edges[num_edges].v_idx2[1] = poly.faces[f].v_idx[0][1];
          edges[num_edges].valid = true;
          num_edges++;
        }

        poly.faces[f].valid = false;
      }

      // Remove duplicate edges (edges shared by two removed faces)
      for (int i = 0; i < num_edges; i++) {
        if (!edges[i].valid) continue;

        for (int j = i + 1; j < num_edges; j++) {
          if (!edges[j].valid) continue;

          // Check if same edge (either direction)
          if ((edges[i].v1 == edges[j].v1 && edges[i].v2 == edges[j].v2) ||
            (edges[i].v1 == edges[j].v2 && edges[i].v2 == edges[j].v1)) {
            edges[i].valid = false;
            edges[j].valid = false;
          }
        }
      }

      // Create new faces from horizon edges
      for (int i = 0; i < num_edges; i++) {
        if (!edges[i].valid) continue;

        // Find next available face slot
        int new_face_idx = -1;
        
        for (int j = 0; j < MAX_EPA_FACES; j++) {
          if (!poly.faces[j].valid) {
            new_face_idx = j;
            break;
          }
        }

        if (new_face_idx < 0 || new_face_idx >= MAX_EPA_FACES) break;

        // Create new face: edge horizon vertices + new vertex
        poly.faces[new_face_idx].v[0] = edges[i].v1;
        poly.faces[new_face_idx].v[1] = edges[i].v2;
        poly.faces[new_face_idx].v[2] = new_vertex_id;

        poly.faces[new_face_idx].v_idx[0][0] = edges[i].v_idx1[0];
        poly.faces[new_face_idx].v_idx[0][1] = edges[i].v_idx1[1];
        poly.faces[new_face_idx].v_idx[1][0] = edges[i].v_idx2[0];
        poly.faces[new_face_idx].v_idx[1][1] = edges[i].v_idx2[1];
        poly.faces[new_face_idx].v_idx[2][0] = new_vertex_idx[0];
        poly.faces[new_face_idx].v_idx[2][1] = new_vertex_idx[1];

        poly.faces[new_face_idx].valid = true;

        // Check winding and fix if necessary
        gkFloat* fv0 = poly.vertices[poly.faces[new_face_idx].v[0]];
        gkFloat* fv1 = poly.vertices[poly.faces[new_face_idx].v[1]];
        gkFloat* fv2 = poly.vertices[poly.faces[new_face_idx].v[2]];

        gkFloat fe0[3], fe1[3], fnormal[3];

        for (int c = 0; c < 3; c++) {
          fe0[c] = fv1[c] - fv0[c];
          fe1[c] = fv2[c] - fv0[c];
        }
        crossProduct(fe0, fe1, fnormal);

        // If normal points toward centroid flip winding
        gkFloat to_cent[3];
        for (int c = 0; c < 3; c++) to_cent[c] = centroid[c] - fv0[c];
        if (dotProduct(fnormal, to_cent) > 0) {
          // Swap v[1] and v[2]
          int tmp_v = poly.faces[new_face_idx].v[1];
          poly.faces[new_face_idx].v[1] = poly.faces[new_face_idx].v[2];
          poly.faces[new_face_idx].v[2] = tmp_v;

          int tmp_idx0 = poly.faces[new_face_idx].v_idx[1][0];
          int tmp_idx1 = poly.faces[new_face_idx].v_idx[1][1];
          poly.faces[new_face_idx].v_idx[1][0] = poly.faces[new_face_idx].v_idx[2][0];
          poly.faces[new_face_idx].v_idx[1][1] = poly.faces[new_face_idx].v_idx[2][1];
          poly.faces[new_face_idx].v_idx[2][0] = tmp_idx0;
          poly.faces[new_face_idx].v_idx[2][1] = tmp_idx1;
        }

        // Update max face index
        if (new_face_idx >= poly.max_face_index) {
          poly.max_face_index = new_face_idx + 1;
        }
      }
  }

  // If we exited due to max iterations, recompute closest face and use it
  if (iteration >= max_iterations) {
    // Find closest face and compute result
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      compute_face_normal_distance(&poly, i);
    }

    int closest_face = -1;
    gkFloat closest_distance = 1e10f;
    for (int i = 0; i < poly.max_face_index; ++i) {
      if (!poly.faces[i].valid) continue;
      if (poly.faces[i].distance >= 0.0f && poly.faces[i].distance < closest_distance) {
        closest_distance = poly.faces[i].distance;
        closest_face = i;
      }
    }

    if (closest_face >= 0) {
      EPAFace* closest = &poly.faces[closest_face];
      gkFloat a0, a1, a2;
      compute_barycentric_origin(poly.vertices[closest->v[0]],
                                 poly.vertices[closest->v[1]],
                                 poly.vertices[closest->v[2]], &a0, &a1, &a2);
      for (int i = 0; i < 3; i++) {
        simplex->witnesses[0][i] = getCoord(bd1, closest->v_idx[0][0], i) * a0
                    + getCoord(bd1, closest->v_idx[1][0], i) * a1
                    + getCoord(bd1, closest->v_idx[2][0], i) * a2;
        simplex->witnesses[1][i] = getCoord(bd2, closest->v_idx[0][1], i) * a0
                    + getCoord(bd2, closest->v_idx[1][1], i) * a1
                    + getCoord(bd2, closest->v_idx[2][1], i) * a2;
        contact_normal[i] = closest->normal[i];
      }
      *distance = -closest_distance;
    }
  }
}