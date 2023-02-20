LOGIN_URL=https://auth.ceda.ac.uk/account/signin/
PASSWORD=${CEDA_PASSWORD}
USERNAME=kehlert
COOKIE_FILE=cookie.txt
rm ${COOKIE_FILE}
TOKEN=`wget ${LOGIN_URL} -O- --save-cookies ${COOKIE_FILE} --keep-session-cookies \
  --server-response --no-check-certificate \
  | grep csrfmiddlewaretoken | sed -r 's/.*value="(.*)".*/\1/'`
POST="csrfmiddlewaretoken=${TOKEN}&username=${USERNAME}&password=${PASSWORD}"
wget ${LOGIN_URL} -O- --load-cookies ${COOKIE_FILE} --save-cookies ${COOKIE_FILE} \
  --keep-session-cookies --no-check-certificate --server-response \
  --post-data "$POST" >/dev/null
wget -e robots=off --mirror --no-parent -r --load-cookies cookie.txt https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-rain-obs/dataset-version-202207/
