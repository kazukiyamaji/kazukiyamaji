console.log("hello  world?");
image_name = [];
noise_list = [];
search_result = [];
original = 0;
url="//"+window.location.host 
console.log(url+"")
function toBlob(base64) {
  var bin = atob(base64.replace(/^.*,/, ''));
  var buffer = new Uint8Array(bin.length);
  for (var i = 0; i < bin.length; i++) {
      buffer[i] = bin.charCodeAt(i);
  }
  // Blobを作成
  try{
      var blob = new Blob([buffer.buffer], {
          type: 'image/png'
      });
  }catch (e){
      return false;
  }
  return blob;
}
$("#generator").on("submit", function (e) {
  keyword = $("#keyword").val();
  e.preventDefault();
  image_name = [];
  data = { keyword: keyword };
  data = JSON.stringify(data);
  $.ajax({
    url: url+"/display1/",
    type: "POST",
    dataType: "json",
    contentType: "application/json",
    data: data,
  }).done(function (response) {
    for (i = 0; i <= 8; i++) {
      image_str = response.image["image" + String(i)].image;
      
      st = "#load_image" + String(i);
      $(st).attr("src", "data:image/png;base64," + image_str);
      $(st).off();
      $(st).on("click", function () {
        tmp = Number(this.id[10]);
        console.log(tmp);
        $("#number1").val(Number(this.id[10]));
      });
      document.getElementById("btn1").removeAttribute("disabled", true);
      document.getElementById("btn2").setAttribute("disabled", true);
      document.getElementById("btn3").setAttribute("disabled", true);
      image_name.push(response.image["image" + String(i)].filename);
    }
    search_result = response.result;
  });
});
$("#get_picture").on("submit", function (e) {
  page = Number($("#page").val());
  e.preventDefault();
  l = [];
  for (var i = 9 * page; i < 9 * (page + 1); i++) {
    l.push(search_result[i]);
  }
  data = { numbers: l };
  data = JSON.stringify(data);
  $.ajax({
    url: url+"/get_picture1/",
    type: "POST",
    dataType: "json",
    contentType: "application/json",
    data: data,
  }).done(function (response) {
    for (i = 0; i <= 8; i++) {
      image_str = response.image["image" + String(i)].image;
      st = "#load_image" + String(i);
      $(st).attr("src", "data:image/png;base64," + image_str);
      $(st).off();
      $(st).on("click", function () {
        tmp = Number(this.id[10]);
        console.log(tmp);
        $("#number1").val(Number(this.id[10]));
        document.getElementById("btn2").removeAttribute("disabled", true);
      });
      //document.getElementById("btn1").setAttribute("disabled", true);
      document.getElementById("btn2").setAttribute("disabled", true);
      document.getElementById("btn3").setAttribute("disabled", true);
      image_name.push(response.image["image" + String(i)].filename);
    }
  });
});
$("#invert").on("submit", function (e) {
  page = Number($("#page").val());
  num = Number($("#number1").val());
  e.preventDefault();
  pid = 9 * page + num;
  id = search_result[pid];
  console.log(id);
  $("#notice").text("Invert中、しばらくお待ちください")
  if ($("#number1").val() != "") {
    data = { id: id };
    data = JSON.stringify(data);
    $.ajax({
      url: url+"/invert1/",
      type: "POST",
      dataType: "json",
      contentType: "application/json",
      data: data,
    }).done(function (response) {
      noise_list = [];
      for (i = 0; i <= 8; i++) {
        image_str = response.image["image" + String(i)]["image"];
        st = "#load_image" + String(i);
        original = response.image["image" + String(i)]["vec"];
        noise_list.push(response.image["image" + String(i)]["vec"]);
        $(st).attr("src", "data:image/png;base64," + image_str);
        $(st).off();
        $(st).on("click", function () {
          tmp = Number(this.id[10]);
          $("#number2").val(Number(this.id[10]));
          document.getElementById("btn3").removeAttribute("disabled", true);
        });
        document.getElementById("btn1").setAttribute("disabled", true);
        document.getElementById("btn2").setAttribute("disabled", true);
        document.getElementById("btn3").setAttribute("disabled", true);
        $("#notice").text("Invert終了、好きな画像を選んでください")
      }
    });
  }
});
$("#select").on("submit", function (e) {
  num = $("#number2").val();
  e.preventDefault();
  if ($("#number2").val() != "") {
    console.log(noise_list);
    if ($("#same:checked").val() == "on") {
      vec = original;
      $("#notice").text("もう一度生成しています、しばらくお待ちください")
    } else {
      vec = noise_list[num];
      $("#notice").text("選択した画像をもとに生成しています、しばらくお待ちください")
    }
    data = {
      vec: vec,
      label: $("#number2").val(),
      mode: $("#mode").val(),
      variation: $("#variation").val(),
      direction: $("#direction").val(),
    };
    data = JSON.stringify(data);
    $.ajax({
      url: url+"/cycle_generation1/",
      type: "POST",
      dataType: "json",
      contentType: "application/json",
      data: data,
    }).done(function (response) {
      noise_list = [];
      original = response.original;
      console.log(original);
      for (i = 0; i <= 8; i++) {
        image_str = response.image["image" + String(i)]["image"];
        st = "#load_image" + String(i);
        noise_list.push(response.image["image" + String(i)]["vec"]);
        $(st).attr("src", "data:image/png;base64," + image_str);
        
        $(st).off();
        $(st).on("click", function () {
          tmp = Number(this.id[10]);
          $("#number2").val(Number(this.id[10]));
        });
      }
      $("#notice").text("生成終了、好きな画像を選んでください")
    });
  }
});

$("#download").on("submit", function (e) {
  num = $("#number2").val();
  console.log("download")
  e.preventDefault();
  if ($("#number2").val() != "") {
    console.log(noise_list);
    vec = noise_list[num];
    data = {
      vec: vec
    };
    data = JSON.stringify(data);
    $.ajax({
      url: url+"/download/",
      type: "POST",
      dataType: "json",
      contentType: "application/json",
      data: data,
    }).done(function (response) {
      image_str=response.image
      b=toBlob(image_str)
      console.log(b)
      let link = document.createElement('a')
      link.href = window.URL.createObjectURL(b)
      link.download = 'download-filename.png'
      link.click()
    });
  }
});