function orig_img=reverse_wind(wind_img)
    orig_img=zeros(304,304,72,'uint8');
    for i_img=1:72
        pointer=1;
        for i_row=1:38:304
            for i_col=1:38:304
                current_wind=wind_img(:,:,pointer,i_img);
                orig_img(i_row:(i_row+37),i_col:(i_col+37),i_img)=current_wind;
                pointer=pointer+1;
            end
        end
    end

end