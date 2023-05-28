from shiny import *
import pandas as pd
from shiny.types import FileInfo
import geopandas as gpd
from geopandas import io
import geodatasets
import matplotlib.pyplot as plt
import ipyleaflet as L
from ipyleaflet  import Map, GeoData, basemaps, LayersControl
from shinywidgets import *
import numpy as np
import seaborn as sns
from matplotlib import rcParams
import matplotlib.axes as ax
import shinyswatch
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta


import subprocess

package_name = "skimpy"
package_2 = "shinyswatch"

# Use pip to install the package
subprocess.check_call(["pip", "install", package_name])
subprocess.check_call(["pip", "install", package_2])



app_ui = ui.page_fluid(
    bokeh_dependency(),
    shinyswatch.theme.pulse(),

    ui.navset_tab(

        ui.nav("Welcome!", 
               
ui.tags.style(
        """
        .app-col {
            border-radius: 5px;
            padding: 8px;
            margin-top: 5px;
            margin-bottom: 5px;
        }
        """
    ),
    ui.br(),
    ui.h1({"style": "text-align: center;"}, "STAT 390: Data Science Project"),
    ui.h5({"style": "text-align: center;"}, "Team Data Girlies: Matthew Phi, Cathy Kim, Angelina Jaglinski, Ada Zhong"),
    ui.row(
        ui.column(
            12,
            ui.div(
                {"class": "app-col"},
                ui.p({"style": "text-align: center;"},
                    """
                    Welcome!
                    """,
                ),
                ui.p(
                    """
                    As part of our STAT 390 course this quarter, our team created an interactive dashboard that will allow the user to interact with the dataset in a meaningful way. Our hope is that this will be a useful tool for easy analyses on the programs offered combined with various variables of interest. 
                    """,
                ),
            ),
        )
    ),
    ui.output_image("image"),
    
)
                
                
                ,

    ui.nav("Uploading CSV File", 
           ui.br(),
           ui.br(),
           ui.input_file("file1", 
                
                "Choose a file to upload and clean:", accept=".csv"),
                ui.download_button("download1", "Download Cleaned CSV")),

    ui.nav("Visualization", 
           
           ui.layout_sidebar(
               ui.panel_sidebar(
                   ui.input_radio_buttons(
        "rb",
        "Choose one:",
        {
            "num_programs": "Number of Programs",
            "min_age": "Minimum Age",
            "capacity": "Capacity",
            "km_to_bus_stop" : "Kilometers to Bus Stop"
        },
    ),
                   ui.input_slider("x3", "Age Range Slider", value=(9, 14), min=0, max=25),
    ui.output_text("value"),
    ui.input_checkbox("somevalue", "All Community Areas?", False),
    ui.input_checkbox("somevalue3", "Focus Areas?", False),
    ui.input_selectize("cluster", "Community Area", choices = [], multiple=True),
    ui.input_checkbox("somevalue2", "All Categories?", False),
    ui.input_selectize("Programs", "Program Category", choices = [], multiple=True),
    ui.output_text("program"),
               ),
               ui.panel_main(
                   ui.br(),
                   ui.br(),
                ui.output_plot("plot")
               )
           )

               
               
               
               
               ),




    ui.nav("Maps", 
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_radio_buttons(
                "framework",
                "Choose a Visualization:",
                [

                    "Public_Transportation",
                    "Age_Range",
                    "Capacity",
                    #"Accessibility_Index"
                ],
            ),

        ),
        ui.panel_main(
            ui.output_ui("figure"),
        ),
    ),



           ),
        ui.nav("Accessibility Index", 
    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_radio_buttons(
                "framework2",
                "",
                [
                    "Accessibility_Index"
                ],
            ),
            ui.input_slider("trans", "Public Transportation Weight", min=0, max=100, value = 20),
            ui.input_slider("cap", "Capacity Weight", min=0, max=100, value = 20),
            ui.input_slider("price", "Price Weight", min=0, max=100, value = 20),
            ui.input_slider("food", "Free Food Weight", min=0, max=100, value = 20),
            ui.input_slider("safe", "Safety Weight", min=0, max=100, value = 20),

        ),
        ui.panel_main(
            ui.output_ui("figure2"),
        ),
    ),



           ),







    ui.nav("Prediction", 
           
           ui.layout_sidebar(
               ui.panel_sidebar(
                   ui.input_slider("pred_age", "Age", min=0, max=24, value = 12),
                ui.input_selectize("pred_cluster", "Community Area", choices = [], multiple=True),
                ui.input_selectize("pred_program", "Program Category", choices = [], multiple=True),
                ui.input_date("pred_date", "Month:", value="2024-01-12", format="mm/yyyy")
               ),
               ui.panel_main(
                   ui.br(),
                   ui.br(),
                ui.output_plot("plot2")
               )
           )

               
               
               
               
               ),





    ),


    )




def server(input: Inputs, output: Outputs, session: Session):

    @output
    @render.image
    def image():
        from pathlib import Path
        dir = Path(__file__).resolve().parent
        img: ImgData = {"src": str(dir / "logos.png"), "width": "450px"}
        return img

    @reactive.Calc 
    @reactive.Effect
    def _():
        if ts_data() is None:
            return
        else:
            data = ts_data()
            ui.update_selectize(
            "pred_cluster",
            choices=sorted(data["geographic_cluster_name"].unique().tolist()),
            server=False,
            )

            ui.update_selectize(
            "pred_program",
            choices=sorted(data["category_name"].unique().tolist()),
            server=False,
            )



    @output
    @render.plot
    def plot2():
        if ts_data() is None or input.pred_cluster() is None or input.pred_program() is None:
            return 
        else:
            df = ts_data()
            data = pd.DataFrame()
            df = df[(input.pred_age() <= df["max_age"]) & (input.pred_age() >= df["min_age"])]
        #filter = data[(data["geographic_cluster_name"] == str(input.cluster()[0]) ) | (data["geographic_cluster_name"] == str(input.cluster()[1]) )]
        #df = pd.concat([df, filter])
        #return round(df["min_age"].mean(),2)
        #filter = data[data["geographic_cluster_name"] == "IRVING PARK"]
            for i in range(len(input.pred_cluster())):
                x = df[df["geographic_cluster_name"] == str(input.pred_cluster()[i])]
                data = pd.concat([data, x])
            df = data
            data = pd.DataFrame()
            for i in range(len(input.pred_program())):
                x = df[df["category_name"] == str(input.pred_program()[i])]
                data = pd.concat([data, x])
            df = data
            #df = df[(input.pred_age() <= df["max_age"]) & (input.pred_age() >= df["min_age"])]
            sample_date = pd.to_datetime(input.pred_date())
            def count_programs_by_features(data):
    # Extract the month and year from the start_date column
                data["start_date"] = pd.to_datetime(data["start_date"])
                data["month"] = data["start_date"].dt.month
                data["year"] = data["start_date"].dt.year
                features_with_month = ["month", "year"]
                program_counts = pd.DataFrame(data.groupby(features_with_month).size().reset_index(name="count"))
                return program_counts
            
            if df.empty:
                return
            else:
                program_counts = count_programs_by_features(df)
            # space
            #space
            #space
            #space
            #space
                program_counts['date'] = pd.to_datetime(program_counts[['year', 'month']].assign(day=1))

                model = ARIMA(program_counts["count"], order=(1, 0, 0))
                model_fit = model.fit()
                start_date = max(program_counts["date"])
                delta = relativedelta(sample_date, start_date)
                future_periods = delta.years * 12 + delta.months
                forecast = model_fit.forecast(steps=future_periods)
                grouped_data = program_counts.groupby("date")
                plt.figure(figsize=(12, 6))
            

                final_graph = plt.scatter(program_counts["date"], program_counts["count"])
                plt.xlabel("Time")
                plt.ylabel("Count")
                plt.title("Program Counts Over Time")
                plt.xticks(rotation=45)
# Plot the predicted value
                plt.plot(sample_date,forecast.iloc[-1], marker="o", color="red") #forecasted prediction
            

                return final_graph








    

    @reactive.Calc
    @output(id="figure2")
    @render.ui
    def _():
        return output_widget(input.framework2())
    

    @reactive.Calc
    def new_data():
        import json
        from bokeh.io import show
        from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
        from bokeh.layouts import column, row
        from bokeh.palettes import brewer
        from bokeh.plotting import figure

        from bokeh.io import output_notebook, show, output_file
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.tile_providers import Vendors
        from bokeh.palettes import PRGn, RdYlGn, all_palettes
        from bokeh.transform import linear_cmap,factor_cmap
        from bokeh.layouts import row, column
        from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, BasicTicker
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import geodatasets
        from bokeh.io import output_file, show
        from bokeh.models import ColumnDataSource, GMapOptions
        from bokeh.plotting import gmap
        data2 = ts_data()
        chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
        chicago['community'] = chicago['community'].apply(str.lower)
        gdf = gpd.GeoDataFrame(data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude), crs="EPSG:4326")
        gdf_grouped = gdf.groupby('geographic_cluster_name')['km_to_bus_stop'].mean().reset_index()
        gdf_grouped = pd.DataFrame(gdf_grouped)
        gdf_grouped['geographic_cluster_name'] = gdf_grouped['geographic_cluster_name'].apply(str.lower)
        gdf_grouped['community'] = gdf_grouped['geographic_cluster_name']
        chicago_df = pd.merge(chicago, gdf_grouped, on = 'community')
        chicago_df = chicago_df.drop(columns = ["NID", "POP2010", "POP2000", "POPCH", "POPPERCH", "popplus", "popneg", "geographic_cluster_name"])
        chicago_df = chicago_df.to_crs({'init': 'epsg:3857'})
        chicago_df["km_to_bus_stop"] = chicago_df["km_to_bus_stop"].round(3)

        # making capacity data

        data2 = ts_data()
        gdf = gpd.GeoDataFrame(data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude), crs="EPSG:4326")
        gdf_grouped = gdf.groupby('geographic_cluster_name')['capacity'].mean().reset_index()
        gdf_grouped = pd.DataFrame(gdf_grouped)
        gdf_grouped['geographic_cluster_name'] = gdf_grouped['geographic_cluster_name'].apply(str.lower)
        gdf_grouped['community'] = gdf_grouped['geographic_cluster_name']
        chicago_df2 = pd.merge(chicago, gdf_grouped, on = 'community')
        chicago_df2["capacity_per_capita"] = chicago_df2["capacity"] / chicago_df2["POP2010"]
        chicago_df2 = chicago_df2.drop(columns = ["NID", "POP2010", "POP2000", "POPCH", "POPPERCH", "popplus", "popneg", "geographic_cluster_name", "geometry"])

        
        #making Safety data
        safe = pd.read_csv("neighborhood_safety.csv")
        safe['community'] = safe['Name'].apply(str.lower)
        safe["safety2"] = safe["safety"]/100



        # making food data
        df = ts_data()
        chicago_df3 = df
        chicago_df3['food_score'] = np.where(chicago_df3['program_provides_free_food'] == True, 1, 0)
        chicago_df3 = chicago_df3.groupby("geographic_cluster_name")["food_score"].mean().reset_index()
        chicago_df3["community"] = chicago_df3["geographic_cluster_name"].apply(str.lower)



        # making price data
        data = ts_data()
        Conditions = [
    (data['program_price'] == "Free"),
    (data['program_price'] == "$50 or Less"),
    (data['program_price'] == "More Than $50"),
    (data['program_price'] == "Unknown")
]
        Categories = [1,0.5,0.25,0]
        data['price_score'] = np.select(Conditions, Categories)
        data = data.groupby("geographic_cluster_name")["price_score"].mean().reset_index()
        data["community"] = data["geographic_cluster_name"].apply(str.lower)

        # Merging Everything
        final1 = pd.merge(chicago_df, chicago_df2, on = 'community')
        final2 = pd.merge(final1, safe, on = "community")
        final3 = pd.merge(final2, chicago_df3, on = "community")
        final = pd.merge(final3, data, on = "community")
        
        final["cap_score"] = (final["capacity_per_capita"]- final["capacity_per_capita"].min())/ (final["capacity_per_capita"].max()-final["capacity_per_capita"].min() )
        final["trans_score"] = (final["km_to_bus_stop"].max() - final["km_to_bus_stop"]) / (final["km_to_bus_stop"].max() - final["km_to_bus_stop"].min())
        final["price_score2"] = final["price_score"]


        trans_weight = float(input.trans())
        cap_weight = float(input.cap())
        price_weight = float(input.price())
        food_weight = float(input.food())
        safe_weight = float(input.safe())

        total_weight = trans_weight + cap_weight + price_weight + food_weight + safe_weight 
        #total_weight = 1


        final["total_weight"] = (final["trans_score"] * trans_weight + final["cap_score"] * cap_weight + final["price_score2"] * price_weight + final["safety2"] * safe_weight + final["food_score"] * food_weight) / (total_weight)
        #final["total_weight"] = (final["km_to_bus_stop"] * 20 + final["capacity"] * 20 + final["price_score"] * 20 + final["safety2"] * 20) / (total_weight*100)



        return final

    @output
    @render.table
    @reactive.Calc
    def stuff():
        data = new_data()
        return data

    

   
   
    @reactive.Calc
    @output(id="figure")
    @render.ui
    def _():
        return output_widget(input.framework())
    

    @reactive.Calc
    @output(id="Accessibility_Index")
    @render_widget
    def _():
        import json
        from bokeh.io import show
        from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
        from bokeh.layouts import column, row
        from bokeh.palettes import brewer
        from bokeh.plotting import figure

        from bokeh.io import output_notebook, show, output_file
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.tile_providers import Vendors
        from bokeh.palettes import PRGn, RdYlGn, all_palettes
        from bokeh.transform import linear_cmap,factor_cmap
        from bokeh.layouts import row, column
        from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, BasicTicker
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import geodatasets
        from bokeh.io import output_file, show
        from bokeh.models import ColumnDataSource, GMapOptions
        from bokeh.plotting import gmap
        



        if ts_data is None:
            return
        else:


            data = new_data()

            geosource = GeoJSONDataSource(geojson = data.to_json())



            TOOLTIPS = [
    ("Community", "@community"),
    ("Accessibility Score", "@total_weight")
    ]
            tools = 'wheel_zoom,pan,reset'

            palette = all_palettes['Viridis'][10]

            color_mapper = LinearColorMapper(palette = palette, low = data['total_weight'].min(), high = data['total_weight'].max())

            p = figure(title = "Accessibility", width=700, 
           height=760, toolbar_location='right', tooltips=TOOLTIPS, tools=tools,
          x_axis_type="mercator", y_axis_type="mercator")
            p.add_tile(Vendors.OSM)

# This Vendors.OSM is the Open Street Map tile that basically puts the OSM map over the figure --> this is what's 
# not working on this code but works on the other one

            p.patches('xs','ys', source = geosource,fill_color = {'field' :'total_weight', 'transform' : color_mapper},
          line_color = 'black', line_width = 1, fill_alpha = 0.7)
# the p.patches is how you include the chicago clusters into the figure



            color_bar = ColorBar(color_mapper=color_mapper,  ticker= BasicTicker(),
                formatter = NumeralTickFormatter(format='0.0[0000]'), 
            label_standoff = 20, width=8, location=(0,0), padding = 5, title = "Accessibility Index", major_tick_line_color = "black",major_tick_out = 8, major_label_text_font_size = "15px")
# Set color_bar location
            p.add_layout(color_bar, 'right') 
            p.title.text_align = 'center'
            p.title.text_font_size = '18pt'


            return p



    @reactive.Calc
    @output(id="Public_Transportation")
    @render_widget
    def _():
        import json
        from bokeh.io import show
        from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
        from bokeh.layouts import column, row
        from bokeh.palettes import brewer
        from bokeh.plotting import figure

        from bokeh.io import output_notebook, show, output_file
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.tile_providers import Vendors
        from bokeh.palettes import PRGn, RdYlGn, all_palettes
        from bokeh.transform import linear_cmap,factor_cmap
        from bokeh.layouts import row, column
        from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, BasicTicker
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import geodatasets
        from bokeh.io import output_file, show
        from bokeh.models import ColumnDataSource, GMapOptions
        from bokeh.plotting import gmap
        



        if ts_data is None:
            return
        else:


            data2 = ts_data()
        #data2.loc[data2.geographic_cluster_name == 'Back of the Yards', 'geographic_cluster_name'] = "NEW CITY"
        #data2.loc[data2.geographic_cluster_name == 'Little Village', 'geographic_cluster_name'] = "SOUTH LAWNDALE"
        #data2.loc[data2.geographic_cluster_name == 'Bronzeville/South Lakefront', 'geographic_cluster_name'] = "OAKLAND"
            chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
            chicago['community'] = chicago['community'].apply(str.lower)
            gdf = gpd.GeoDataFrame(data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude), crs="EPSG:4326")
            gdf_grouped = gdf.groupby('geographic_cluster_name')['km_to_bus_stop'].mean().reset_index()
            gdf_grouped = pd.DataFrame(gdf_grouped)
            gdf_grouped['geographic_cluster_name'] = gdf_grouped['geographic_cluster_name'].apply(str.lower)
            gdf_grouped['community'] = gdf_grouped['geographic_cluster_name']
            chicago_df = pd.merge(chicago, gdf_grouped, on = 'community')
            chicago_df = chicago_df.drop(columns = ["NID", "POP2010", "POP2000", "POPCH", "POPPERCH", "popplus", "popneg", "geographic_cluster_name"])
            chicago_df2 = chicago_df.to_crs({'init': 'epsg:3857'})
            chicago_df2["km_to_bus_stop"] = chicago_df2["km_to_bus_stop"].round(3)
            geosource = GeoJSONDataSource(geojson = chicago_df2.to_json())



            TOOLTIPS = [
    ("Community", "@community"),
    ("Km to Bus Stop", "@km_to_bus_stop")
    ]
            tools = 'wheel_zoom,pan,reset'

            palette = all_palettes['Viridis'][10]

            color_mapper = LinearColorMapper(palette = palette, low = chicago_df2['km_to_bus_stop'].max(), high = chicago_df2['km_to_bus_stop'].min())

            p = figure(title = "Average Distance to a Bus Stop (Km)", width=700, 
           height=760, toolbar_location='right', tooltips=TOOLTIPS, tools=tools,
          x_axis_type="mercator", y_axis_type="mercator")
            p.add_tile(Vendors.OSM)

# This Vendors.OSM is the Open Street Map tile that basically puts the OSM map over the figure --> this is what's 
# not working on this code but works on the other one

            p.patches('xs','ys', source = geosource,fill_color = {'field' :'km_to_bus_stop', 'transform' : color_mapper},
          line_color = 'black', line_width = 1, fill_alpha = 0.7)
# the p.patches is how you include the chicago clusters into the figure



            color_bar = ColorBar(color_mapper=color_mapper,  ticker= BasicTicker(),
                formatter = NumeralTickFormatter(format='0.0[0000]'), 
            label_standoff = 20, width=8, location=(0,0), padding = 5, title = "Avg Km to a Bus Stop", major_tick_line_color = "black",major_tick_out = 8, major_label_text_font_size = "15px")
# Set color_bar location
            p.add_layout(color_bar, 'right') 
            p.title.text_align = 'center'
            p.title.text_font_size = '18pt'


            return p

    @reactive.Calc
    @output(id="Age_Range")
    @render_widget
    def _():
        import json
        from bokeh.io import show
        from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
        from bokeh.layouts import column, row
        from bokeh.palettes import brewer
        from bokeh.plotting import figure

        from bokeh.io import output_notebook, show, output_file
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.tile_providers import Vendors
        from bokeh.palettes import PRGn, RdYlGn, all_palettes
        from bokeh.transform import linear_cmap,factor_cmap
        from bokeh.layouts import row, column
        from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, BasicTicker
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import geodatasets
        from bokeh.io import output_file, show
        from bokeh.models import ColumnDataSource, GMapOptions
        from bokeh.plotting import gmap
        



        if ts_data is None:
            return
        else:


            data2 = ts_data()
            data2["age_range"] = data2["max_age"] - data2["min_age"]
        #data2.loc[data2.geographic_cluster_name == 'Back of the Yards', 'geographic_cluster_name'] = "NEW CITY"
        #data2.loc[data2.geographic_cluster_name == 'Little Village', 'geographic_cluster_name'] = "SOUTH LAWNDALE"
        #data2.loc[data2.geographic_cluster_name == 'Bronzeville/South Lakefront', 'geographic_cluster_name'] = "OAKLAND"
            chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
            chicago['community'] = chicago['community'].apply(str.lower)
            gdf = gpd.GeoDataFrame(data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude), crs="EPSG:4326")
            gdf_grouped = gdf.groupby('geographic_cluster_name')['age_range'].mean().reset_index()
            gdf_grouped = pd.DataFrame(gdf_grouped)
            gdf_grouped['geographic_cluster_name'] = gdf_grouped['geographic_cluster_name'].apply(str.lower)
            gdf_grouped['community'] = gdf_grouped['geographic_cluster_name']
            chicago_df = pd.merge(chicago, gdf_grouped, on = 'community')
            chicago_df = chicago_df.drop(columns = ["NID", "POP2010", "POP2000", "POPCH", "POPPERCH", "popplus", "popneg", "geographic_cluster_name"])
            chicago_df2 = chicago_df.to_crs({'init': 'epsg:3857'})
            chicago_df2["age_range"] = chicago_df2["age_range"].round(3)
            geosource = GeoJSONDataSource(geojson = chicago_df2.to_json())



            TOOLTIPS = [
    ("Community", "@community"),
    ("Age Range", "@age_range")
    ]
            tools = 'wheel_zoom,pan,reset'

            palette = all_palettes['Viridis'][10]

            color_mapper = LinearColorMapper(palette = palette, low = chicago_df2['age_range'].min(), high = chicago_df2['age_range'].max())

            p = figure(title = "Average Age Range in Chicago", width=700, 
           height=760, toolbar_location='right', tooltips=TOOLTIPS, tools=tools,
          x_axis_type="mercator", y_axis_type="mercator")
            p.add_tile(Vendors.OSM)

# This Vendors.OSM is the Open Street Map tile that basically puts the OSM map over the figure --> this is what's 
# not working on this code but works on the other one

            p.patches('xs','ys', source = geosource,fill_color = {'field' :'age_range', 'transform' : color_mapper},
          line_color = 'black', line_width = 1, fill_alpha = 0.7)
# the p.patches is how you include the chicago clusters into the figure



            color_bar = ColorBar(color_mapper=color_mapper,  ticker= BasicTicker(),
                formatter = NumeralTickFormatter(format='0.0[0000]'), 
            label_standoff = 20, width=8, location=(0,0), padding = 5, title = "Age Range", major_tick_line_color = "black",major_tick_out = 8, major_label_text_font_size = "15px")
# Set color_bar location
            p.add_layout(color_bar, 'right')
            p.title.text_align = 'center'
            p.title.text_font_size = '18pt' 


            return p
        

    @output(id="Capacity")
    @render_widget
    def _():
        import json
        from bokeh.io import show
        from bokeh.models import (CDSView, ColorBar, ColumnDataSource,
                          CustomJS, CustomJSFilter, 
                          GeoJSONDataSource, HoverTool,
                          LinearColorMapper, Slider)
        from bokeh.layouts import column, row
        from bokeh.palettes import brewer
        from bokeh.plotting import figure

        from bokeh.io import output_notebook, show, output_file
        from bokeh.plotting import figure, ColumnDataSource
        from bokeh.tile_providers import Vendors
        from bokeh.palettes import PRGn, RdYlGn, all_palettes
        from bokeh.transform import linear_cmap,factor_cmap
        from bokeh.layouts import row, column
        from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, NumeralTickFormatter, BasicTicker
        import numpy as np
        import pandas as pd
        import geopandas as gpd
        import geodatasets
        from bokeh.io import output_file, show
        from bokeh.models import ColumnDataSource, GMapOptions
        from bokeh.plotting import gmap
        



        if ts_data is None:
            return
        else:


            data2 = ts_data()
            data2["age_range"] = data2["max_age"] - data2["min_age"]
        #data2.loc[data2.geographic_cluster_name == 'Back of the Yards', 'geographic_cluster_name'] = "NEW CITY"
        #data2.loc[data2.geographic_cluster_name == 'Little Village', 'geographic_cluster_name'] = "SOUTH LAWNDALE"
        #data2.loc[data2.geographic_cluster_name == 'Bronzeville/South Lakefront', 'geographic_cluster_name'] = "OAKLAND"
            chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
            chicago['community'] = chicago['community'].apply(str.lower)
            gdf = gpd.GeoDataFrame(data2, geometry=gpd.points_from_xy(data2.longitude, data2.latitude), crs="EPSG:4326")
            gdf_grouped = gdf.groupby('geographic_cluster_name')['capacity'].mean().reset_index()
            gdf_grouped = pd.DataFrame(gdf_grouped)
            gdf_grouped['geographic_cluster_name'] = gdf_grouped['geographic_cluster_name'].apply(str.lower)
            gdf_grouped['community'] = gdf_grouped['geographic_cluster_name']
            chicago_df = pd.merge(chicago, gdf_grouped, on = 'community')
            chicago_df["capacity_per_capita"] = chicago_df["capacity"] / chicago_df["POP2010"]
            chicago_df = chicago_df.drop(columns = ["NID", "POP2010", "POP2000", "POPCH", "POPPERCH", "popplus", "popneg", "geographic_cluster_name"])
            chicago_df2 = chicago_df.to_crs({'init': 'epsg:3857'})
            geosource = GeoJSONDataSource(geojson = chicago_df2.to_json())



            TOOLTIPS = [
    ("Community", "@community"),
    ("Capacity", "@capacity"),
    ("Capacity Per Capita", "@capacity_per_capita")
    ]
            tools = 'wheel_zoom,pan,reset'

            palette = all_palettes['Viridis'][10]

            color_mapper = LinearColorMapper(palette = palette, low = chicago_df2['capacity_per_capita'].min(), high = chicago_df2['capacity_per_capita'].max())

            p = figure(title = "Average Capacity per Capita in Chicago", width=700, 
           height=760, toolbar_location='right', tooltips=TOOLTIPS, tools=tools,
          x_axis_type="mercator", y_axis_type="mercator")
            p.add_tile(Vendors.OSM)

# This Vendors.OSM is the Open Street Map tile that basically puts the OSM map over the figure --> this is what's 
# not working on this code but works on the other one

            p.patches('xs','ys', source = geosource,fill_color = {'field' :'capacity_per_capita', 'transform' : color_mapper},
          line_color = 'black', line_width = 1, fill_alpha = 0.7)
# the p.patches is how you include the chicago clusters into the figure



            color_bar = ColorBar(color_mapper=color_mapper,  ticker= BasicTicker(),
                formatter = NumeralTickFormatter(format='0.0[0000]'), 
            label_standoff = 20, width=8, location=(0,0), padding = 5, title = "Capacity per Capita", major_tick_line_color = "black",major_tick_out = 8, major_label_text_font_size = "15px")
# Set color_bar location
            p.add_layout(color_bar, 'right') 

            p.title.text_align = 'center'
            p.title.text_font_size = '18pt'


            return p





        


    
    @reactive.Calc
    def ts_data():
        f: list[FileInfo] = input.file1()
        if f is None:
            return
        else:
            if f[0]["name"] == "data_cleaned.csv":
                import pandas as pd
                df = pd.read_csv(f[0]["datapath"])
                df.loc[df.geographic_cluster_name == 'Back of the Yards', 'geographic_cluster_name'] = "NEW CITY"
                df.loc[df.geographic_cluster_name == 'Little Village', 'geographic_cluster_name'] = "SOUTH LAWNDALE"
                df.loc[df.geographic_cluster_name == 'Bronzeville/South Lakefront', 'geographic_cluster_name'] = "OAKLAND"
                df = df.dropna(axis=0, subset=['geographic_cluster_name'])
                return df
            else:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                import string
                import re
                from skimpy import clean_columns
                from math import radians
                from sklearn.metrics.pairwise import haversine_distances

                df = pd.read_csv(f[0]["datapath"], sep='\t')
                bus_data = pd.read_csv('CTA_BusStops_Cleaned.csv')
                def clean_data(data, bus_data):
    # Clean column names
                    data = clean_columns(data)
    
    # Remove all programs where min age > 25
                    data = data[data["min_age"] < 25]
    
    # Clean category names (program types)
    # Make lowercase
                    data.category_name = data.category_name.apply(lambda x: x.lower() if isinstance(x, str) else x)

                    def remove_punctuation(text):
                        if isinstance(text, str):
                            return text.translate(str.maketrans('', '', string.punctuation))
                        else:
                            return text

                    def replace_spaces_with_underscore(text):
                        if isinstance(text, str):
            # Replace one or more whitespace characters with a single underscore
                            return re.sub(r'\s+', '_', text)
                        else:
                            return text

                    def remove_trailing_underscores(text):
                        if isinstance(text, str):
            # Remove any trailing underscores
                            text = text.rstrip('_')
                            return text
                        else:
                            return text
        
    # Remove punctuation, replace spaces with underscores, remove trailing underscores for category names
                    data.category_name = data.category_name.apply(remove_punctuation)
                    data.category_name = data.category_name.apply(replace_spaces_with_underscore)
                    data.category_name = data.category_name.apply(remove_trailing_underscores)
    
    # Clean state + city
                    data["state"] = data["state"].replace("Illinois", "IL")
                    data = data[data.city == "Chicago"]
    
    # Add age range col
                    data["age_range"] = data.max_age - data.min_age
                    data = data[~data["latitude"].isna() & ~data["longitude"].isna() ]
    # Add distance to nearest bus stop col
                    bus_coords = bus_data[['latitude', 'longitude']].to_numpy()
                    mcmf_coords = data[['latitude', 'longitude']].to_numpy()
    
                    mcmf_in_radians = np.array([[radians(float(x)) for x in coord] for coord in mcmf_coords])
                    bus_in_radians = np.array([[radians(float(x)) for x in coord] for coord in bus_coords])
     
                    dists = haversine_distances(mcmf_in_radians, bus_in_radians) # Calculate haversine dists then convert to km
                    dists_km = dists*6371 # multiply by radius of Earth
    
                    min_dist = np.nanmin(dists_km, axis=1) # Minimum distances to each bus stop

                    data['km_to_bus_stop'] = min_dist
    
                    return data
                data = clean_data(df, bus_data)
                data.loc[data.geographic_cluster_name == 'Back of the Yards', 'geographic_cluster_name'] = "NEW CITY"
                data.loc[data.geographic_cluster_name == 'Little Village', 'geographic_cluster_name'] = "SOUTH LAWNDALE"
                data.loc[data.geographic_cluster_name == 'Bronzeville/South Lakefront', 'geographic_cluster_name'] = "OAKLAND"
                data = data.dropna(axis=0, subset=['geographic_cluster_name'])
                return data



    
    @session.download(filename="data_cleaned.csv")
    def download1():
        yield ts_data().to_csv()

    

    @reactive.Calc
    @output
    @render.plot
    def plot():

        if ts_data() is None:
            return
        else:
            df = ts_data()
            if input.somevalue() == True:
                df = ts_data()
            elif input.somevalue3() == True:
                df = df[(df.geographic_cluster_name == "AUSTIN") | 
                         (df.geographic_cluster_name == "NORTH LAWNDALE") |
                         (df.geographic_cluster_name == "HUMBOLDT PARK") |
                         (df.geographic_cluster_name == "EAST GARFIELD PARK") |
                         (df.geographic_cluster_name == "ENGLEWOOD") |
                         (df.geographic_cluster_name == "AUBURN GRESHAM") |
                         (df.geographic_cluster_name == "EAST GARFIELD PARK") |
                         (df.geographic_cluster_name == "WEST GARFIELD PARK") |
                         (df.geographic_cluster_name == "ROSELAND") |
                         (df.geographic_cluster_name == "GREATER GRAND CROSSING") |
                         (df.geographic_cluster_name == "WEST ENGLEWOOD") |
                         (df.geographic_cluster_name == "SOUTH SHORE") |
                         (df.geographic_cluster_name == "NEW CITY") |
                         (df.geographic_cluster_name == "CHICAGO LAWN") | 
                         (df.geographic_cluster_name == "SOUTH LAWNDALE") |
                         (df.geographic_cluster_name == "WEST PULLMAN")]
            else:
                data = pd.DataFrame()
        #filter = data[(data["geographic_cluster_name"] == str(input.cluster()[0]) ) | (data["geographic_cluster_name"] == str(input.cluster()[1]) )]
        #df = pd.concat([df, filter])
        #return round(df["min_age"].mean(),2)
        #filter = data[data["geographic_cluster_name"] == "IRVING PARK"]
                for i in range(len(input.cluster())):
                    x = df[df["geographic_cluster_name"] == str(input.cluster()[i])]
                    data = pd.concat([data, x])
                df = data

            if ts_data() is None:
                return
            else:
                if input.somevalue2() == True:
                    df = df
                elif input.Programs() is None:
                    return
                else:
                    data = pd.DataFrame()
                    for i in range(len(input.Programs())):
                        x = df[df["category_name"] == str(input.Programs()[i])]
                        data = pd.concat([data, x])
                    df = data


            

            if input.rb() == "num_programs":
                title = "Number of Programs"
            elif input.rb() == "min_age":
                title = "Avg. Minimum Age"
            elif input.rb() == "capacity":
                title = "Avg. Capacity"
            elif input.rb() == "km_to_bus_stop":
                title = "Avg. Kilometers to Bus Stop"
            else:
                title = "Placeholder"


            

            if df.empty:
                return 
            elif input.somevalue() == False and input.somevalue3() == False and input.cluster() is None or input.somevalue2() == False and input.Programs() is None:
                return
                
            else:
                if input.x3()[1] == 25:
                    df = df[df["min_age"] >= input.x3()[0]]
                elif input.x3()[0] == input.x3()[1]:
                    df = df[df["min_age"] == input.x3()[0]]
                else:
                    df = df[df["min_age"]>= input.x3()[0]]
                    df = df[df["max_age"]<= input.x3()[1]]
                

                if input.rb() == "num_programs":
                    rcParams['figure.figsize'] = 30,8
                    final_data_agg = df.groupby(["geographic_cluster_name", "category_name"]).count().reset_index()
                    graph = sns.barplot(x=final_data_agg.geographic_cluster_name, y=final_data_agg.iloc[:,2], hue=final_data_agg.category_name, data=final_data_agg)
                
                    graph.set_xticklabels(graph.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

                #plt.xticks(fontsize = 8, ha = "right", rotation_mode = "anchor")
                #ax.set_xticks(ha = "right", rotation_mode = "anchor")



                    plt.xlabel("Neighborhood", fontsize=15)
                    plt.ylabel(title, fontsize=15)
                    plt.title("Metrics for Chicago Programs", fontsize=22)
                    plt.legend(title="Program Type", bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0)
                    return graph

                else:
                    rcParams['figure.figsize'] = 30,8
                    final_data_agg = df.groupby(["geographic_cluster_name", "category_name"]).agg(selected_metric=(input.rb(), "mean")).reset_index()
                    graph = sns.barplot(x=final_data_agg.geographic_cluster_name, y=final_data_agg.iloc[:,2], hue=final_data_agg.category_name, data=final_data_agg)
                
                    graph.set_xticklabels(graph.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

                #plt.xticks(fontsize = 8, ha = "right", rotation_mode = "anchor")
                #ax.set_xticks(ha = "right", rotation_mode = "anchor")



                    plt.xlabel("Neighborhood", fontsize=15)
                    plt.ylabel(title, fontsize=15)
                    plt.title("Metrics for Chicago Programs", fontsize=22)
                    plt.legend(title="Program Type", bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0)
                    return graph
                



        





    

    @output
    @render.text
    def value():
        if input.x3()[0] > 24:
            return "There are no programs above 24!"
        else:
            return
            #return "You shouldn't be here"
            #return "You choose: " + str(input.x3()[0])



    @reactive.Calc  
    @output
    @render.text
    def program():
        if ts_data() is None:
            return
        else:
            df = ts_data()
            if input.somevalue2() == True:
                df = ts_data()
            else:
                data = pd.DataFrame()
                for i in range(len(input.Programs())):
                    x = df[df["category_name"] == str(input.Programs()[i])]
                    data = pd.concat([data, x])
                df = data
        
            if df.empty:
                return "Please include some Categories!"
    

    @reactive.Calc 
    @reactive.Effect
    def _():
        if ts_data() is None:
            return
        else:
            data = ts_data()
            ui.update_selectize(
            "cluster",
            choices=sorted(data["geographic_cluster_name"].unique().tolist()),
            server=False,
            )

            ui.update_selectize(
            "Programs",
            choices=sorted(data["category_name"].unique().tolist()),
            server=False,
            )







app = App(app_ui, server)