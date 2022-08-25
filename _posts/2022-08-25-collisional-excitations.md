---
title: "Energy Equipartition in Collisional Excitations"
date: 2022-08-25
categories:
  - blog
tags:
  - physics
  - python
  - statistical mechanics
---

When a system is in thermodynamic equilibrium, its energy is shared equally among all of its degrees of freedom. This is known as the equipartition theorem. A simple example to demonstrate this is a collection of atoms, each of which has two internal states: one ground state and one excited state. If all atoms begin in their ground states, collisions can exite the internal states of the atoms, converting kinetic energy to excitational energy. Similarly, if more atoms are in the excited state, they can be de-excited by collisions, converting this internal excitation energy into larger recoil velocities for the two colliding atoms.

This is demonstrated in the movie below. The atoms all start in the blue (ground) state, and collisions cause some of the atoms to transition into the red (excited) state. Over time, the amount of kinetic energy and the amount of internal excitational energy equilibrates, resulting in roughly equal excitation and kinetic temperatures, shown in the graph on the right.

<div class="myvideo">
   <video  style="display:block; width:100%; height:auto;" controls loop="loop">
     <source src="{{ site.baseurl }}/assets/videos/collisions.mp4" type="video/mp4" />
   </video>
</div>

The code for generating this animation is available on my Github [here](https://github.com/MatthewQuenneville/collisional-excitations).
