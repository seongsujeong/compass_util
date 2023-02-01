import numpy as np
from osgeo import gdal

ds = gdal.Open("NETCDF:gslc/gslc_x5_y5.h5://science/LSAR/GSLC/grids/frequencyA/HH", gdal.GA_ReadOnly)
geotrans = ds.GetGeoTransform()


x0 = geotrans[0]
dx = geotrans[1]
x0 += dx/2.
y0 = geotrans[3]
dy = -5.0 #geotrans[5]
y0 += dy/2.

print("geo transform read with gdal:")
print(x0, dx, y0, dy)
print("*******************")

lats = [3.177088785]
lons = [-54.579586258]

epsg = 32621

#for ii in range(len(lats)):
for ii, lat in enumerate(lats):
    #lat = lats[ii]
    lon = lons[ii]

    x,y = lat_lon_to_utm(lat, lon, 0.0, epsg)
    print(f"UTM coordinats CR (x,y): {x}, {y}")
    x_index = (x - x0)/dx
    y_index = (y - y0)/dy

    print(f"x ,y index : {x_index}, {y_index}")

    data = ds.ReadAsArray()

    data = data[75:125, 75:125]
    ovs_factor = 128
    data_ref_ov =  pti.oversample(data, ovs_factor)
    amp = np.abs(data_ref_ov)
    print("np.argmax(amp)", np.argmax(amp))
    t_y = int(np.argmax(amp)/amp.shape[1])
    t_x = np.argmax(amp) - t_y*amp.shape[1]
    print(f"estimated target x, y: {t_x/ovs_factor}, {t_y/ovs_factor}")

    t_y = t_y/ovs_factor + 75
    t_x = t_x/ovs_factor + 75

    err_x = t_x - x_index
    err_y = t_y - y_index

    print(f"x error [pixels]: {err_x}")
    print(f"y error [pixels]: {err_y}")

    err_x *= dx
    err_y *= dy
    print(f"x error [meters]: {err_x}")
    print(f"y error [meters]: {err_y}")

data_all = ds.ReadAsArray()

plt.imshow(20*np.log10(np.abs(data_all)**2), cmap="gray", vmin = -100, vmax=25)
plt.plot(x_index, y_index, "ro", ms=4)
plt.show()