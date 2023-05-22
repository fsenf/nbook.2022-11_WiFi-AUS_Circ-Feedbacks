;*************************************************
; sat_aod_2d_plot.ncl
;
; Concepts illustrated:
;   - Calculating a two-dimensional correlation in time
;   - Reordering an array
;   - Copying attributes from one variable to another
;   - Copying coordinate arrays from one variable to another
;
;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
begin


wks  = gsn_open_wks("png","Fig2")              ; open a ps plot;


;************************************************
; open file and read in variable
;***********************************************


   dmo1="~/data/wifi-aus/postproc/aot-comparison/"




expname="_injtrp+1_1000_pcb"


fname2="wifiaus_ham_long_nu_gfas2020"+expname+"_ao"

fnamebase="wifiaus_ham_long_nu_gfas2020_injtp+1_0000_pcb_ao"



fname1="wifiaus_ham_long_nu_gfas2020"

fmo01=fnamebase+"_202001.01_radm.nc"
fmo02=fnamebase+"_202002.01_radm.nc"
fmo03=fnamebase+"_202003.01_radm.nc"



fmo21=fname2+"_202001.01_radm.nc"
fmo22=fname2+"_202002.01_radm.nc"
fmo23=fname2+"_202003.01_radm.nc"




 in01  = addfile(dmo1+fmo01,"r")
 in02  = addfile(dmo1+fmo02,"r")
 in03  = addfile(dmo1+fmo03,"r")
in21  = addfile(dmo1+fmo21,"r")
in22 = addfile(dmo1+fmo22,"r")
in23 = addfile(dmo1+fmo23,"r")


;-----

tmp01 = in01->TAU_2D_550nm(0,:,:)
tmp02 = in02->TAU_2D_550nm(0,:,:)
tmp03 = in03->TAU_2D_550nm(0,:,:)
tmp21 = in21->TAU_2D_550nm(0,:,:)
tmp22 = in22->TAU_2D_550nm(0,:,:)
tmp23 = in23->TAU_2D_550nm(0,:,:)

data01 = tmp01(:,:)
data02 = tmp02(:,:)
data03 = tmp03(:,:)
data21 = tmp21(:,:)-tmp01(:,:)
data22 = tmp22(:,:)-tmp02(:,:)
data23 = tmp23(:,:)-tmp03(:,:)


;----








  lat = in02->lat
  lon = in02->lon
  nlat=dimsizes(lat)
  nlon=dimsizes(lon)





  
;***********************************************
; plot dust AOT
;************************************************

gsn_define_colormap(wks,"MPL_YlOrBr")            ; choose colormap
gsn_define_colormap(wks,"hotres")
;gsn_define_colormap(wks,"MPL_BuPu")
;gsn_define_colormap(wks,"MPL_PuRd")
gsn_define_colormap(wks,"MPL_Pink");<-





;gsn_define_colormap(wks,"MPL_RdPu")
;gsn_define_colormap(wks,"MPL_YlOrRd")
gsn_define_colormap(wks,"MPL_gist_heat") ;<-
gsn_reverse_colormap(wks)

nc = NhlNewColor(wks,0.8,0.8,0.8)



res                      = True                ; make plot mods

;res@cnFillColors =  (/130,5,13,34,50,75,100,129/) orig

;;res@cnFillColors =  (/130,3,6,15,40,70,90,129/) ;heatmap
res@cnFillColors =  (/5,18,35,60,85,105/)

;res@cnFillColors =  (/130,5,20,40,60,80,110,129/) ;pink(schlamm)

; setup map resources
res = True
;  res@mpProjection           = "polar"
res@gsnPolar                                  = "SH"
;  res@mpLimitMode            = "Corners"
;  res@mpLeftCornerLatF       = lat(nlat-1)
;res@mpLeftCornerLonF       = -178; lon(0)

;res@mpRightCornerLatF      = lat(0)
;res@mpRightCornerLonF      = 178;lon(nlon-1)


; no re-projection (we are on the "native" grid)
res@sfXArray               = lon*360./358.125
res@sfYArray               = lat

res@gsnMaximize            = False             ; maxmize plot in frame
res@gsnDraw                = False
res@gsnFrame               = False
res@gsnSpreadColors        = True             ; use full color map
res@gsnAddCyclic           = False

res@mpDataBaseVersion      = "MediumRes"      ; use finer database
res@mpOutlineBoundarySets  = "National"
;  res@pmTickMarkDisplayMode  = "Always"
res@mpGeophysicalLineThicknessF = 2.5

;res@gsnTickMarksOn = False
res@gsnPolarLabelSpacing = 60
res@gsnPanelYWhiteSpacePercent = 20
res@gsnPolarLabelFontHeightF= .02


res@cnFillOn             = True                ; turn on color

;---These two resources not needed in NCL V6.1.0
res@gsnSpreadColors      = True                ; use full colormap
;res@gsnSpreadColorStart = 1
;res@gsnSpreadColorEnd = -1

;---This resource defaults to True in NCL V6.1.0
  res@lbLabelAutoStride    = True                ; automatic lb label stride
res@lbLabelAutoStride    = True                ; automatic lb label stride

res@cnLevelSelectionMode   = "ExplicitLevels"
;res@cnLabelBarEndStyle     = "ExcludeOuterBoxes"
res@lbLabelFontHeightF     = 0.028
res@lbTitleFontHeightF      =  0.033
res@lbLabelBarOn           = True
res@pmLabelBarHeightF = 0.14
res@cnLinesOn            = False               ; turn off contour lines

res@lbTitleString ="AOT (550 nm)"
res@lbTitlePosition = "Bottom"

res@cnLevels = (/0.05,0.1,0.15,0.2,0.25/)
res@cnLevels = res@cnLevels*0.2
;res@cnLevels = res@cnLevels*0.1

res@gsnStringFontHeightF = 0.02



res@lbLabelBarOn           = True


; res@gsnRightString         = "AOD"
;************************************************

res@mpCenterLonF=180


cmap = gsn_retrieve_colormap(wks)

plot=new(6,graphic)



  res@gsnLeftString         = "Jan"
  plot(3) = gsn_csm_contour_map_polar(wks,data21,res)
print(max(data21))

res@gsnLeftString         = "Feb"
plot(4) = gsn_csm_contour_map_polar(wks,data22,res)

res@gsnLeftString         = "Mar"
plot(5) = gsn_csm_contour_map_polar(wks,data23,res)



;print(max(data0(95,0)))
;print(max(data10(95,0))
;print(max(data1(95,0))
;print(max(data2(95,0))
;print(max(data30(95,0))
;print(max(data3(95,0)))



;draw(plot)   ; Drawing the plot draws the attached markers too

    gsn_panel(wks,plot,(/2,3/),False)



end






	
