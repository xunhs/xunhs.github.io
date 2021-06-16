---
layout: post
title: Python库-osmnx&networkx
toc: true
top: false
categories:
  - 收藏
tags:
  - Python库
  - osm
  - osmnx
  - graph
  - networkx
  - road network
abbrlink: 9712dcfa
date: 2020-05-27T09:25:17+00:00
---

> osmnx是下载、处理等osm路网很好用的一个包。networkx是图论学习中很好用的包，两者结合为基于图论的方法研究osm road network提供便利。以下内容为整理两个包常用的功能函数方法。


<!--more-->




### osmnx基础
#### Init
```python
import osmnx as ox
import networkx as nx
%matplotlib inline
ox.config(log_console=True, use_cache=True)
ox.__version__

from pathlib import Path
```

#### Query place
```python
place ="Shenzhen"
G = ox.graph_from_place(place, network_type='drive',which_result=2)
```
- Note parameter `network_type` means specify several different network types(Refer from [overview-osmnx](https://github.com/gboeing/osmnx-examples/blob/master/notebooks/01-overview-osmnx.ipynb)):
  - 'drive' - get drivable public streets (but not service roads)
  - 'drive_service' - get drivable streets, including service roads
  - 'walk' - get all streets and paths that pedestrians can use (this network type ignores one-way directionality)
  - 'bike' - get all streets and paths that cyclists can use
  - 'all' - download all non-private OSM streets and paths (this is the default network type unless you specify a different one)
  - 'all_private' - download all OSM streets and paths, including private-access ones

- private-access road means roads with tag-`access=private`. (Refer from [wiki-osm](https://wiki.openstreetmap.org/wiki/Tag:access%3Dprivate))


#### Plot graph
```python
fig, ax = ox.plot_graph(G, node_size=0, edge_color='w', edge_linewidth=0.2, bgcolor='k')
```
![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200527135418.png)



#### Save/load shapefile & graphml
Refer: [osmnx-examples/05-save-load-networks.ipynb](https://github.com/gboeing/osmnx-examples/blob/master/notebooks/05-save-load-networks.ipynb)
```python
# get a network
# save graph as a shapefile
# save/load graph as a graphml file: this is the best way to save your model
# for subsequent work later

shenzhen_dir = Path('./shenzhen')
shp_dir = Path(shenzhen_dir, 'shp')
graph_fp = Path(shenzhen_dir, 'shenzhen.graphml')

if not shp_dir.exists():
    shp_dir.mkdir()

# shapefile & graphml
ox.save_graph_shapefile(G, filepath=str(shp_dir))
ox.save_graphml(G, filepath=str(graph_fp), gephi=False)

# load graphml
G = ox.load_graphml(graph_fp)
```

#### Calculate basic network indicators
```python
stats = ox.basic_stats(G)
stats
```
out:
```python
{'n': 33121,
 'm': 73564,
 'k_avg': 4.442136408924851,
 'intersection_count': 29210,
 'streets_per_node_avg': 3.0009963467286616,
 'streets_per_node_counts': {0: 0,
  1: 3911,
  2: 126,
  3: 21360,
  4: 7485,
  5: 221,
  6: 18},
 'streets_per_node_proportion': {0: 0.0,
  1: 0.11808218350895203,
  2: 0.0038042329639805562,
  3: 0.6449080643700371,
  4: 0.2259895534555116,
  5: 0.006672503849521451,
  6: 0.0005434618519972223},
 'edge_length_total': 14743317.355999907,
 'edge_length_avg': 200.41484090043917,
 'street_length_total': 10292802.579999937,
 'street_length_avg': 207.1486592336165,
 'street_segments_count': 49688,
 'node_density_km': None,
 'intersection_density_km': None,
 'edge_density_km': None,
 'street_density_km': None,
 'circuity_avg': 1.0846412027749917,
 'self_loop_proportion': 0.0014001413734979066,
 'clean_intersection_count': None,
 'clean_intersection_density_km': None}
```


#### Fast nearest node/edge search with OSMnx
```python
center_point = lat, lng # 注意坐标格式，纬度在前，经度在后

# find the nearest node to some point
center_node = ox.get_nearest_node(G, center_point)


# find the nearest nodes to a set of points
# optionally specify `method` use use a kdtree or balltree index
nearest_nodes = ox.get_nearest_nodes(G, lngs, lats, method='balltree')



# find the nearest edge to some point
nearest_edge = ox.get_nearest_edge(G, center_point)


# find the nearest edges to some set of points
# optionally specify `method` use use a kdtree or balltree index
nearest_edges = ox.get_nearest_edges(G, lngs, lats)
```



### OSMnx functions
#### graph <-> GeoDataFrames
Refer to : [osmnx.utils_graph](https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_graph.graph_from_gdfs)
##### Convert a graph into node and/or edge GeoDataFrames
```python
edges_gdf = ox.graph_to_gdfs(G, nodes=False)
nodes_gdf = ox.graph_to_gdfs(G, edges=False)
edges_gdf.head();nodes_gdf.head()
```
out:  
edges_gdf  
|    |          u |          v |   key |     osmid | name     | highway  | oneway   |   length | geometry  |   lanes |   maxspeed |   ref |   bridge |   tunnel |   access |   width |   landuse |   junction |
|---:|-----------:|-----------:|------:|----------:|:---------|:---------------|:---------|---------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------:|-----------:|------:|---------:|---------:|---------:|--------:|----------:|-----------:|
|  0 | 3377201155 | 3377199760 |     0 | 330725486 | 南同大道 | secondary      | False    |   87.53  | LINESTRING (114.2827267 22.6972648, 114.2818811 22.6971596)  |     nan |        nan |   nan |      nan |      nan |      nan |     nan |       nan |        nan |
|  1 | 3377201155 | 3145194682 |     0 | 330725486 | 南同大道 | secondary      | False    |  284.987 | LINESTRING (114.2827267 22.6972648, 114.2841709 22.6974048, 114.2854842 22.6975733) |     nan |        nan |   nan |      nan |      nan |      nan |     nan |       nan |        nan |
|  2 | 3377201155 | 3377201142 |     0 | 330727857 | nan      | unclassified   | False    |  332.196 | LINESTRING (114.2827267 22.6972648, 114.2825873 22.6983075, 114.2826002 22.6984817, 114.2829494 22.7002143) |     nan |        nan |   nan |      nan |      nan |      nan |     nan |       nan |        nan |
|  3 | 5296488468 | 2720855960 |     0 | 266553759 | nan      | secondary_link | False    |    9.983 | LINESTRING (114.103413 22.7115999, 114.1033684 22.7116797) |     nan |        nan |   nan |      nan |      nan |      nan |     nan |       nan |        nan |
|  4 | 5296488468 | 5296488469 |     0 | 548236871 | 中环大道 | tertiary       | True     | 1143.42  | LINESTRING (114.103413 22.7115999, 114.1037991 22.7116604, 114.1041064 22.711632, 114.1044133 22.7114466, 114.1047399 22.7111653, 114.104999 22.7108329, 114.1053723 22.7105961, 114.1062619 22.7103602, 114.1066166 22.7100907, 114.1067313 22.7096768, 114.1066358 22.7091382, 114.106247 22.7087397, 114.1057681 22.7082979, 114.1053296 22.7078727, 114.1050831 22.7075478, 114.1050484 22.7069286, 114.1051519 22.7064617, 114.105374 22.7060621, 114.1054614 22.7054678, 114.1054724 22.7048434, 114.1054836 22.7042894, 114.1054666 22.7039499) |     nan |        nan |   nan |      nan |      nan |      nan |     nan |       nan |        nan |
nodes_gdf  
|            |       y |       x |      osmid |   highway |   ref | geometry                       |
|-----------:|--------:|--------:|-----------:|----------:|------:|:-------------------------------|
| 3377201155 | 22.6973 | 114.283 | 3377201155 |       nan |   nan | POINT (114.2827267 22.6972648) |
| 5296488468 | 22.7116 | 114.103 | 5296488468 |       nan |   nan | POINT (114.103413 22.7115999)  |
| 5296488469 | 22.7039 | 114.105 | 5296488469 |       nan |   nan | POINT (114.1054666 22.7039499) |
| 5296488477 | 22.704  | 114.103 | 5296488477 |       nan |   nan | POINT (114.1033273 22.7039622) |
| 2728263712 | 22.6701 | 114.138 | 2728263712 |       nan |   nan | POINT (114.1384942 22.6700615) |

- `osmnx.utils_graph.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)`
  - G (networkx.MultiDiGraph) – input graph
  - nodes (bool) – if True, convert graph nodes to a GeoDataFrame and return it
  - edges (bool) – if True, convert graph edges to a GeoDataFrame and return it
  - node_geometry (bool) – if True, create a geometry column from node x and y data
  - fill_edge_geometry (bool) – if True, fill in missing edge geometry fields using origin and destination nodes

##### Convert node and edge GeoDataFrames into a MultiDiGraph
```python
G = ox.graph_from_gdfs(gdf_edges=edges_gdf, gdf_nodes=nodes_gdf)
```
- 'osmnx.utils_graph.graph_from_gdfs(gdf_nodes, gdf_edges)'
  - gdf_nodes (geopandas.GeoDataFrame) – GeoDataFrame of graph nodes
  - gdf_edges (geopandas.GeoDataFrame) – GeoDataFrame of graph edges




#### 路径规划
##### Basic routing by distance
Pick two nodes. Then find the shortest path between origin and destination, using weight='length' to find the shortest path by minimizing distance traveled (otherwise it treats each edge as weight=1).
```python
# find the shortest path (by distance) between these nodes then plot it
orig = list(G)[0] # node id 3377201155
dest = list(G)[-1] # node id 3377201149
route = nx.shortest_path(G, orig, dest) # [3377201155, 3145194682, ...]
fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
```
The routing correctly handles one-way streets:
```python
orig_pnt = (22.67304, 114.0133) # 注意坐标格式，纬度在前，经度在后
dest_pnt = (22.589, 114.0973)
origin_node = ox.get_nearest_node(G, orig_pnt)
destination_node = ox.get_nearest_node(G, dest_pnt)
route = nx.shortest_path(G, origin_node, destination_node)
fig, ax = ox.plot_graph_route(G, route, route_linewidth=6, node_size=0, bgcolor='k')
```


![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200527142311.png)

- Note `shortest_path(G, source=None, target=None, weight=None, method='dijkstra')`
  - G (NetworkX graph)
  - source (node, optional) – Starting node for path. If not specified, compute shortest paths for each possible starting node.
  - target (node, optional) – Ending node for path. If not specified, compute shortest paths to all possible nodes.
  - weight (None or string, optional (default = None)) – If None, every edge has weight/distance/cost 1. If a string, use this edge attribute as the edge weight. Any edge attribute not present defaults to 1.
  - method (string, optional (default = ‘dijkstra’)) – The algorithm to use to compute the path. Supported options: ‘dijkstra’, ‘bellman-ford’. Other inputs produce a ValueError. If weight is None, unweighted graph methods are used, and this suggestion is ignored.
  - Return: path – All returned paths include both the source and target in the path.If the source and target are both specified, return a single list of nodes in a shortest path from the source to the target.If only the source is specified, return a dictionary keyed by targets with a list of nodes in a shortest path from the source to one of the targets.If only the target is specified, return a dictionary keyed by sources with a list of nodes in a shortest path from one of the sources to the target.If neither the source nor target are specified return a dictionary of dictionaries with path[source][target]=[list of nodes in path].


##### Routing by travel speeds and times
add_edge_speeds: The add_edge_speeds function add edge speeds (km per hour) to graph as new speed_kph edge attributes. Imputes free-flow travel speeds for all edges based on mean maxspeed value of edges, per highway type. This mean-imputation can obviously be imprecise, and the caller can override it by passing in hwy_speeds and/or fallback arguments that correspond to local speed limit standards. 
```python
# impute speed on all edges missing data
G = ox.add_edge_speeds(G)

# calculate travel time (seconds) for all edges
G = ox.add_edge_travel_times(G)

# see mean speed/time values by road type
edges = ox.graph_to_gdfs(G, nodes=False)
edges['highway'] = edges['highway'].astype(str)
edges.groupby('highway')[['length', 'speed_kph', 'travel_time']].mean().round(1) # round(1)取小数点后一位数


# same thing again, but this time pass in a few default speed values (km/hour)
# to fill in edges with missing `maxspeed` from OSM
hwy_speeds = {'residential': 35,
              'secondary': 50,
              'tertiary': 60}
G = ox.add_edge_speeds(G, hwy_speeds)
G = ox.add_edge_travel_times(G)

```
out:  

| highway                          |   length |   speed_kph |   travel_time |
|:---------------------------------|---------:|------------:|--------------:|
| ['living_street', 'residential'] |    534.4 |        20   |          96.2 |
| ['motorway', 'motorway_link']    |   2978.7 |        94.1 |         113.9 |
| ['motorway', 'secondary']        |   3239.3 |        94.1 |         123.9 |
| ['motorway', 'trunk_link']       |    631.4 |        30   |          75.8 |
| ['motorway_link', 'secondary']   |    856.6 |        46.7 |          66.1 |  

Routing by travel speeds and times  
```python
# calculate two routes by minimizing travel distance vs travel time
orig = list(G)[1]
dest = list(G)[-1]
route1 = nx.shortest_path(G, orig, dest, weight='length')
route2 = nx.shortest_path(G, orig, dest, weight='travel_time')

# compare the two routes
route1_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route1, 'length')))
route2_length = int(sum(ox.utils_graph.get_route_edge_attributes(G, route2, 'length')))
route1_time = int(sum(ox.utils_graph.get_route_edge_attributes(G, route1, 'travel_time')))
route2_time = int(sum(ox.utils_graph.get_route_edge_attributes(G, route2, 'travel_time')))
print('Route 1 is', route1_length, 'meters and takes', route1_time, 'seconds.')
print('Route 2 is', route2_length, 'meters and takes', route2_time, 'seconds.')
```
out:  
Route 1 is 28753 meters and takes 1787 seconds.   
Route 2 is 36221 meters and takes 1647 seconds.   

```python
# pick route colors
c1 = 'r' #length
c2 = 'b' #travel_time
rc1 = [c1] * (len(route1) - 1)
rc2 = [c2] * (len(route2) - 1)
rc = rc1 + rc2
nc = [c1, c1, c2, c2]

# plot the routes
fig, ax = ox.plot_graph_routes(G, [route1, route2], route_color=rc, route_linewidth=6,
                               orig_dest_node_color=nc, node_size=0, bgcolor='k')
```

![](https://gitee.com/xunhs/xunhs/raw/master/pics/2020/summer/20200527145917.png)
The blue route minimizes travel time, and is thus longer but faster than the red route.

***

![](https://cdn.jsdelivr.net/gh/xunhs/image_host/history/ethan.imfast.io/imgs/2020/05/photo-of-people-riding-on-pickup-truck-4388158.jpg)

<!-- Functions: -->
<!-- 插入音乐 -->
<!-- Refer: https://github.com/MoePlayer/hexo-tag-aplayer -->
<!-- Demo -->
<!-- \{\% meting "558290126" "netease" "song" "autoplay" "mutex:false" "preload:none" "theme:#ad7a86"\%\} -->


<!-- 插入视频 -->
<!-- Bilibili -->
<!-- 加上 id="bilibili-player" 设置css用 -->
<!-- Demo -->
<!-- <iframe id="bilibili-player" src="//player.bilibili.com/player.html?aid=57056321&bvid=BV16x411d7Rb&cid=99643214&page=1&as_wide=1&high_quality=1&danmaku=0" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true" sandbox="allow-top-navigation allow-same-origin allow-forms allow-scripts"> </iframe> -->