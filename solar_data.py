import requests


if __name__ == '__main__':
    url = "http://developer.nrel.gov/api/solar/nsrdb_0512_download.json?api_key=Zye9o6TuKrYAPyGAnpGZayoABAPbRjzB4fYkhiot"

    payload = "names=2012&leap_day=false&interval=60&utc=false&full_name=Rupa%2BKV&email=rkurinch%40caltech.edu&affiliation=NREL&mailing_list=true&reason=Academic&attributes=dhi%2Cdni%2Cwind_speed_10m_nwp%2Csurface_air_temperature_nwp&wkt=MULTIPOINT(-106.22%2032.9741%2C-106.18%2032.9741%2C-106.1%2032.9741)"

    headers = {
        'content-type': "application/x-www-form-urlencoded",
        'cache-control': "no-cache"
    }

    response = requests.request("POST", url, data=payload, headers=headers)

    print(response.text)